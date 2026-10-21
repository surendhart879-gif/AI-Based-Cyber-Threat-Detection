import json
import os
import matplotlib.pyplot as plt
import os

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

from preprocess import prepare_features
from behavior import analyze_behavior
from realtime import RealtimeMonitor, list_interfaces
from flask import send_file, jsonify
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

app = Flask(__name__)
app.secret_key = "cyber-threat-detection"

# Holds the most recent analysis so /download_report can build a PDF
# from it. Fine for a single-user local Flask app; for multi-user
# deployments this should move to a per-session store instead.
last_analysis = {"results": None, "filename": None}

BLOCKED_IPS_PATH = "blocked_ips.json"


def load_blocked_ips():
    """Read the simulated 'blocked IPs' list from disk."""
    if not os.path.exists(BLOCKED_IPS_PATH):
        return set()
    with open(BLOCKED_IPS_PATH, encoding="utf-8") as f:
        try:
            return set(json.load(f))
        except json.JSONDecodeError:
            return set()


def save_blocked_ips(ips):
    with open(BLOCKED_IPS_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(ips), f, indent=2)


# In-memory set, kept in sync with blocked_ips.json. This is a
# SIMULATED block list for demo purposes only — it does not touch
# any real firewall or network device.
blocked_ips = load_blocked_ips()

# Realtime ("online") monitor — built once the trained model is
# available. Lives for the whole app process, started/stopped from
# the /realtime page.
_rt_model, _rt_feature_columns, _rt_meta = None, None, None
monitor = None


def get_monitor():
    """Lazily build the RealtimeMonitor once the model is loadable."""
    global monitor
    if monitor is None:
        model, feature_columns, _ = load_model()
        if model is None:
            return None
        monitor = RealtimeMonitor(model, feature_columns)
    return monitor

UPLOAD_FOLDER = "uploads"
MODEL_PATH = os.path.join("model", "threat_model.pkl")
FEATURES_PATH = os.path.join("model", "feature_columns.pkl")
META_PATH = os.path.join("model", "model_meta.json")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)

    return model, feature_columns, meta


def read_csv_safely(filepath_or_buffer):
    """Try UTF-8 first, then fall back to common alternate encodings
    used by CICIDS2017 / Windows-exported CSV files."""
    for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
        try:
            if hasattr(filepath_or_buffer, "seek"):
                filepath_or_buffer.seek(0)
            return pd.read_csv(filepath_or_buffer, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # Last resort: ignore undecodable bytes rather than failing outright
    if hasattr(filepath_or_buffer, "seek"):
        filepath_or_buffer.seek(0)
    return pd.read_csv(filepath_or_buffer, encoding="utf-8", encoding_errors="ignore")


def analyze_csv(filepath):
    model, feature_columns, meta = load_model()

    if model is None:
        return {"error": "Model not found. Run train_model.py first."}

    data = read_csv_safely(filepath)

    features, labels = prepare_features(data, feature_columns)

    predictions = model.predict(features)

    # ---- NEW: behavior-based analysis (Layer 2) ----
    # Groups by source IP (+ time window if a timestamp column exists)
    # and flags entities whose request volume or threat ratio is
    # statistically abnormal. Falls back gracefully if the CSV has
    # no IP-like column.
    behavior_result = analyze_behavior(data, predictions)

    # Create chart

    counts = pd.Series(predictions).value_counts()
    plt.figure(figsize=(9, 7))

    total = counts.sum()

    def autopct_if_big_enough(pct):
        # Hide the percentage text on very thin slices so labels
        # don't pile up on top of each other; the legend still
        # shows the exact label + count for every slice.
        return f"{pct:.1f}%" if pct >= 3 else ""

    wedges, _texts, _autotexts = plt.pie(
        counts.values,
        labels=None,
        autopct=autopct_if_big_enough,
        startangle=90,
        explode=[0.05] * len(counts),
        pctdistance=0.8,
    )

    legend_labels = [f"{label} ({count})" for label, count in zip(counts.index, counts.values)]
    plt.legend(
        wedges,
        legend_labels,
        title="Prediction",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
    )

    plt.title("Threat Distribution")
    plt.axis("equal")
    plt.tight_layout()

    os.makedirs("static/charts", exist_ok=True)

    pie_chart_path = "static/charts/threat_pie.png"
    plt.savefig(pie_chart_path, bbox_inches="tight", dpi=120)
    plt.close()

    threat_counts = counts.to_dict()

    threats = sum(
        count for label, count in threat_counts.items()
        if label != "BENIGN"
    )

    benign = threat_counts.get("BENIGN", 0)

    sample = pd.DataFrame({
        "Prediction": predictions
    })

    if labels is not None:
        sample["Actual"] = labels.values

    sample_preview = sample.head(10).to_html(
        classes="data-table",
        index=False
    )

    result = {
    "total_rows": len(data),
    "threats": int(threats),
    "benign": int(benign),
    "threat_counts": threat_counts,
    "sample_preview": sample_preview,
    "model_meta": meta,
    "pie_chart": "charts/threat_pie.png",
    # ---- NEW: behavior analysis results passed to template ----
    "behavior": behavior_result,
}

    if labels is not None:
        from sklearn.metrics import accuracy_score

        result["upload_accuracy"] = round(
            float(accuracy_score(labels, predictions)), 4
        )

    return result

@app.route("/")
def login():
    return render_template("login.html")


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    return render_template(
        "dashboard.html",
        results=None,
        model_meta=load_model()[2],
        blocked_ips=blocked_ips,
    )

@app.route("/predict", methods=["POST"])
def predict():
    model, feature_columns, meta = load_model()
    file = request.files["file"]

    if not file:
        return "No file selected"

    data = read_csv_safely(file)
    features, _ = prepare_features(data, feature_columns)

    prediction = model.predict(features)

    return {
        "Total Records": len(prediction),
        "Predictions": prediction.tolist()[:20]

    }
@app.route("/block_ip", methods=["POST"])
def block_ip():
    ip = (request.form.get("ip") or "").strip()
    if ip:
        blocked_ips.add(ip)
        save_blocked_ips(blocked_ips)
        flash(f"{ip} added to the blocked list (simulated).")
    return render_template(
        "dashboard.html",
        results=last_analysis["results"],
        model_meta=load_model()[2],
        filename=last_analysis["filename"],
        blocked_ips=blocked_ips,
    )


@app.route("/unblock_ip", methods=["POST"])
def unblock_ip():
    ip = (request.form.get("ip") or "").strip()
    if ip in blocked_ips:
        blocked_ips.discard(ip)
        save_blocked_ips(blocked_ips)
        flash(f"{ip} removed from the blocked list.")
    return render_template(
        "dashboard.html",
        results=last_analysis["results"],
        model_meta=load_model()[2],
        filename=last_analysis["filename"],
        blocked_ips=blocked_ips,
    )


@app.route("/realtime")
def realtime_page():
    mon = get_monitor()
    if mon is None:
        flash("Model not found. Run train_model.py first.")
        return redirect(url_for("dashboard"))

    iface_info = list_interfaces()
    return render_template(
        "realtime.html",
        status=mon.get_status(),
        interfaces=iface_info["interfaces"],
        interface_error=iface_info["error"],
        blocked_ips=blocked_ips,
    )


@app.route("/realtime/start", methods=["POST"])
def realtime_start():
    mon = get_monitor()
    if mon is None:
        flash("Model not found. Run train_model.py first.")
        return redirect(url_for("dashboard"))

    interface = request.form.get("interface", "").strip()
    if not interface:
        flash("Please choose a network interface first.")
        return redirect(url_for("realtime_page"))

    ok, message = mon.start(interface)
    flash(message)
    return redirect(url_for("realtime_page"))


@app.route("/realtime/stop", methods=["POST"])
def realtime_stop():
    mon = get_monitor()
    if mon is None:
        return redirect(url_for("dashboard"))

    ok, message = mon.stop()
    flash(message)
    return redirect(url_for("realtime_page"))


@app.route("/realtime/status")
def realtime_status():
    mon = get_monitor()
    if mon is None:
        return jsonify({"running": False, "error": "Model not loaded."})
    return jsonify(mon.get_status())


@app.route("/download_report")
def download_report():
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("AI-Based Cyber Threat Detection Report", styles["Heading1"]),
        Paragraph("Model: Random Forest", styles["Normal"]),
        Spacer(1, 0.2 * inch),
    ]

    results = last_analysis["results"]
    filename = last_analysis["filename"]

    if results is None:
        elements.append(Paragraph(
            "Status: No file has been analyzed yet in this session. "
            "Upload a CSV and click 'Analyze Threats' first, then "
            "download the report.",
            styles["Normal"],
        ))
    else:
        elements.append(Paragraph(f"Analyzed file: {filename}", styles["Normal"]))
        elements.append(Paragraph("Status: Analysis Completed", styles["Normal"]))
        elements.append(Spacer(1, 0.25 * inch))

        # ---- Summary table ----
        elements.append(Paragraph("Summary", styles["Heading2"]))
        summary_rows = [
            ["Total Rows", str(results.get("total_rows", "-"))],
            ["Threats Detected", str(results.get("threats", "-"))],
            ["Benign Traffic", str(results.get("benign", "-"))],
        ]
        if "upload_accuracy" in results:
            summary_rows.append(
                ["Accuracy on this file", f"{results['upload_accuracy'] * 100:.2f}%"]
            )
        summary_table = Table(summary_rows, colWidths=[2.5 * inch, 2.5 * inch])
        summary_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 0.25 * inch))

        # ---- Threat breakdown table ----
        threat_counts = results.get("threat_counts", {})
        if threat_counts:
            elements.append(Paragraph("Threat Breakdown", styles["Heading2"]))
            breakdown_rows = [["Label", "Count"]] + [
                [str(label), str(count)] for label, count in threat_counts.items()
            ]
            breakdown_table = Table(breakdown_rows, colWidths=[3 * inch, 2 * inch])
            breakdown_table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            elements.append(breakdown_table)
            elements.append(Spacer(1, 0.25 * inch))

        # ---- Behavior analysis summary ----
        behavior = results.get("behavior")
        if behavior:
            elements.append(Paragraph("Behavior Analysis (IP / Time-window)", styles["Heading2"]))
            if behavior.get("enabled"):
                elements.append(Paragraph(
                    f"Window: {behavior.get('window')} | "
                    f"Flagged groups: {behavior.get('flagged_count', 0)}",
                    styles["Normal"],
                ))
                top = behavior.get("top_offenders", [])[:5]
                if top:
                    ip_key = "Source IP" if "Source IP" in top[0] else list(top[0].keys())[0]
                    rows = [["Source IP", "Threat Ratio", "Total Records"]]
                    for row in top:
                        rows.append([
                            str(row.get(ip_key, "-")),
                            f"{row.get('threat_ratio', 0):.2f}",
                            str(row.get("total_records", "-")),
                        ])
                    behavior_table = Table(rows, colWidths=[2 * inch, 1.5 * inch, 1.5 * inch])
                    behavior_table.setStyle(TableStyle([
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]))
                    elements.append(behavior_table)
            else:
                elements.append(Paragraph(
                    f"Skipped: {behavior.get('reason')}", styles["Normal"]
                ))

    pdf_path = os.path.join("static", "Threat_Report.pdf")
    os.makedirs("static", exist_ok=True)
    pdf = SimpleDocTemplate(pdf_path)
    pdf.build(elements)

    return send_file(pdf_path, as_attachment=True, download_name="Threat_Report.pdf")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or file.filename == "":
        flash("Please select a CSV file.")
        return redirect(url_for("dashboard"))

    if not file.filename.lower().endswith(".csv"):
        flash("Only CSV files are supported.")
        return redirect(url_for("dashboard"))

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    try:
        results = analyze_csv(filepath)
    except Exception as exc:
        import traceback
        traceback.print_exc()  # full details will show in the terminal
        flash(f"Could not analyze file: {exc}")
        return redirect(url_for("dashboard"))

    if "error" in results:
        flash(results["error"])
        return redirect(url_for("dashboard"))

    # remember this analysis so /download_report can include it
    last_analysis["results"] = results
    last_analysis["filename"] = file.filename

    return render_template(
        "dashboard.html",
        results=results,
        model_meta=results.get("model_meta"),
        filename=file.filename,
        blocked_ips=blocked_ips,
    )


if __name__ == "__main__":
    app.run(debug=True)