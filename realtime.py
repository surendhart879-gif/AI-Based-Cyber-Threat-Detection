"""
realtime.py

Optional "online" mode: watches live network traffic and feeds it
through the same trained model + behavior analysis already used for
CSV uploads, instead of a pre-saved file.

This module does NOT capture packets itself. Packet capture and
CICIDS-style flow-feature extraction is delegated to the
`cicflowmeter` command-line tool (pip install cicflowmeter), which is
itself built on Scapy. That tool is launched as a subprocess writing
flow rows to a CSV file; this module tails that file, runs each new
flow through prepare_features() + the trained model, and keeps a
rolling summary that the Flask app exposes to the dashboard.

Requirements to actually USE live monitoring on your machine (not
needed just to import this file, and not available inside a
sandboxed/no-network-access environment):
    pip install scapy cicflowmeter
    Windows: install Npcap (https://npcap.com) and run the app as
             Administrator.
    Linux/Mac: run the app with sudo, or grant the python interpreter
             CAP_NET_RAW / CAP_NET_ADMIN capabilities.
"""

import csv
import os
import subprocess
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

from preprocess import prepare_features
from behavior import analyze_behavior

LIVE_CSV_PATH = os.path.join("uploads", "live_flows.csv")
BUFFER_MAXLEN = 2000           # how many recent flows we keep for behavior analysis
BEHAVIOR_REFRESH_EVERY = 5     # re-run behavior analysis every N new flows
POLL_INTERVAL_SECONDS = 2


def _json_safe(value):
    """
    Recursively convert numpy/pandas scalar types (np.int64, np.bool_,
    pd.Timestamp, etc.) into plain Python types so the status dict can
    always be passed straight to Flask's jsonify() without errors.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def list_interfaces():
    """
    Return the network interfaces available for live capture, using
    Scapy. Returns an empty list (with an 'error' message) if Scapy
    isn't installed — the caller can show that as a hint to the user.
    """
    try:
        from scapy.all import get_if_list
        return {"interfaces": get_if_list(), "error": None}
    except Exception as exc:
        return {"interfaces": [], "error": str(exc)}


class RealtimeMonitor:
    """
    Runs `cicflowmeter` as a subprocess against a chosen network
    interface, tails its CSV output as new flows are written, and
    classifies each flow as it arrives using the already-trained
    model — mirroring the batch CSV-upload pipeline but continuously.
    """

    def __init__(self, model, feature_columns):
        self.model = model
        self.feature_columns = feature_columns

        self._process = None
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # recent raw flow dicts (each tagged with __prediction__),
        # used as the rolling window behavior analysis runs over
        self._buffer = deque(maxlen=BUFFER_MAXLEN)

        self.status = {
            "running": False,
            "interface": None,
            "started_at": None,
            "last_updated": None,
            "total_flows": 0,
            "threats": 0,
            "benign": 0,
            "flagged_count": 0,
            "top_offenders": [],
            "behavior_window": None,
            "error": None,
        }

    # ---------------------------------------------------------- public

    def is_running(self):
        with self._lock:
            return self.status["running"]

    def start(self, interface):
        if self.is_running():
            return False, "Live monitoring is already running."

        os.makedirs(os.path.dirname(LIVE_CSV_PATH), exist_ok=True)
        if os.path.exists(LIVE_CSV_PATH):
            try:
                os.remove(LIVE_CSV_PATH)
            except OSError:
                pass

        cmd = ["cicflowmeter", "-i", interface, "-c", LIVE_CSV_PATH]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            return False, (
                "cicflowmeter is not installed or not on PATH. "
                "Run: pip install scapy cicflowmeter"
            )
        except Exception as exc:
            return False, f"Could not start capture: {exc}"

        self._stop_event.clear()
        self._buffer.clear()

        with self._lock:
            self.status.update({
                "running": True,
                "interface": interface,
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "last_updated": None,
                "total_flows": 0,
                "threats": 0,
                "benign": 0,
                "flagged_count": 0,
                "top_offenders": [],
                "error": None,
            })

        self._thread = threading.Thread(target=self._tail_and_predict, daemon=True)
        self._thread.start()
        return True, f"Live monitoring started on '{interface}'."

    def stop(self):
        if not self.is_running():
            return False, "Live monitoring is not running."

        self._stop_event.set()

        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

        with self._lock:
            self.status["running"] = False

        return True, "Live monitoring stopped."

    def get_status(self):
        with self._lock:
            return _json_safe(dict(self.status))

    # --------------------------------------------------------- internal

    def _tail_and_predict(self):
        # Wait for cicflowmeter to create the output file and write
        # at least a header line before we start tailing it.
        waited = 0.0
        while not self._stop_event.is_set():
            if os.path.exists(LIVE_CSV_PATH) and os.path.getsize(LIVE_CSV_PATH) > 0:
                break
            time.sleep(0.5)
            waited += 0.5
            if waited > 30:
                with self._lock:
                    self.status["error"] = (
                        "No flow data received after 30s. Check that the "
                        "selected interface has traffic and that the app "
                        "is running with admin/root privileges."
                    )
                return

        header = None
        offset = 0
        new_rows_since_refresh = 0

        while not self._stop_event.is_set():
            try:
                with open(LIVE_CSV_PATH, "r", encoding="utf-8", newline="") as f:
                    f.seek(offset)
                    new_lines = f.readlines()
                    offset = f.tell()
            except FileNotFoundError:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            if not new_lines:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            if header is None:
                header = next(csv.reader([new_lines[0]]))
                new_lines = new_lines[1:]

            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    values = next(csv.reader([line]))
                    if len(values) != len(header):
                        continue  # partially-written line, skip this cycle
                    row_dict = dict(zip(header, values))
                except Exception:
                    continue

                self._process_flow(row_dict)
                new_rows_since_refresh += 1

            if new_rows_since_refresh >= BEHAVIOR_REFRESH_EVERY:
                self._refresh_behavior()
                new_rows_since_refresh = 0

            with self._lock:
                self.status["last_updated"] = datetime.now().isoformat(timespec="seconds")

            time.sleep(POLL_INTERVAL_SECONDS)

    def _process_flow(self, row_dict):
        try:
            df_row = pd.DataFrame([row_dict])
            features, _ = prepare_features(df_row, self.feature_columns)
            prediction = self.model.predict(features)[0]
        except Exception:
            # Never let one malformed flow row kill the monitoring thread
            prediction = "BENIGN"

        row_dict["__prediction__"] = prediction
        self._buffer.append(row_dict)

        with self._lock:
            self.status["total_flows"] += 1
            if str(prediction) == "BENIGN":
                self.status["benign"] += 1
            else:
                self.status["threats"] += 1

    def _refresh_behavior(self):
        if not self._buffer:
            return

        buf_df = pd.DataFrame(list(self._buffer))
        predictions = buf_df["__prediction__"]
        buf_df = buf_df.drop(columns=["__prediction__"])

        try:
            result = analyze_behavior(buf_df, predictions)
        except Exception:
            return

        with self._lock:
            self.status["flagged_count"] = result.get("flagged_count", 0)
            self.status["top_offenders"] = result.get("top_offenders", [])
            self.status["behavior_window"] = result.get("window")