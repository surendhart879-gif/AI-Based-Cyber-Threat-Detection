"""
Behavior-based analysis layer.

This module groups traffic records by a source entity (IP or similar
identifier) and, if a timestamp column exists, computes time-window
features (request rate, unique destinations, etc.) to flag suspicious
*behavior* patterns — separate from the existing per-row classification.

It is designed to work even when the CSV does not have the "ideal"
columns: it auto-detects likely candidates and skips gracefully if
nothing usable is found.
"""

import pandas as pd

# Candidate column names we might see across different datasets
IP_CANDIDATES = [
    "Source IP", "Src IP", "src_ip", "source_ip", "SourceIP", "srcip"
]
TIME_CANDIDATES = [
    "Timestamp", "timestamp", "Time", "time", "Flow Start", "start_time"
]
DEST_CANDIDATES = [
    "Destination IP", "Dst IP", "dst_ip", "destination_ip", "DestinationIP"
]
PORT_CANDIDATES = [
    "Destination Port", "Dst Port", "dst_port", "destination_port"
]
BYTES_CANDIDATES = [
    "Flow Bytes/s", "Total Length of Fwd Packets", "Bytes", "bytes"
]


def _find_column(df, candidates):
    """Return the first matching column name found in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def analyze_behavior(data: pd.DataFrame, predictions, window: str = "1min"):
    """
    Attempt behavior-based grouping on top of per-row predictions.

    Returns a dict:
        {
            "enabled": bool,          # whether behavior analysis could run
            "reason": str,            # why it was skipped, if disabled
            "window": str,            # time window used (if any)
            "top_offenders": [...],   # list of dicts, worst-behaving IPs
            "flagged_count": int,     # number of IP/window groups flagged
        }
    """
    # CICIDS2017-style files commonly have leading/trailing spaces in
    # column headers (e.g. " Source IP" instead of "Source IP").
    # Work on a copy with cleaned column names so our candidate list
    # matches regardless of that formatting quirk.
    data = data.rename(columns={c: c.strip() for c in data.columns})

    ip_col = _find_column(data, IP_CANDIDATES)
    time_col = _find_column(data, TIME_CANDIDATES)
    dest_col = _find_column(data, DEST_CANDIDATES)
    port_col = _find_column(data, PORT_CANDIDATES)
    bytes_col = _find_column(data, BYTES_CANDIDATES)

    if ip_col is None:
        return {
            "enabled": False,
            "reason": "No source-IP-like column found in the uploaded CSV.",
            "window": None,
            "top_offenders": [],
            "flagged_count": 0,
        }

    df = data.copy()
    df["__prediction__"] = list(predictions)
    df["__is_threat__"] = df["__prediction__"].astype(str) != "BENIGN"

    if time_col is not None:
        # Try to parse timestamps; if it fails, fall back to no-time mode
        try:
            df["__ts__"] = pd.to_datetime(df[time_col], errors="coerce")
            if df["__ts__"].isna().all():
                time_col = None
        except Exception:
            time_col = None

    agg_dict = {"__is_threat__": "sum", "__prediction__": "count"}
    rename_map = {"__is_threat__": "threat_count", "__prediction__": "total_records"}

    if dest_col is not None:
        agg_dict[dest_col] = "nunique"
        rename_map[dest_col] = "unique_destinations"
    if port_col is not None:
        agg_dict[port_col] = "nunique"
        rename_map[port_col] = "unique_ports"
    if bytes_col is not None:
        agg_dict[bytes_col] = "sum"
        rename_map[bytes_col] = "total_bytes"

    if time_col is not None:
        # Group by IP + time window -> per-window behavior features
        grouped = (
            df.set_index("__ts__")
              .groupby([ip_col, pd.Grouper(freq=window)])
              .agg(agg_dict)
              .rename(columns=rename_map)
              .reset_index()
        )
    else:
        # No usable timestamp -> group by IP only (whole-file window)
        grouped = (
            df.groupby(ip_col)
              .agg(agg_dict)
              .rename(columns=rename_map)
              .reset_index()
        )

    grouped["threat_ratio"] = (
        grouped["threat_count"] / grouped["total_records"].replace(0, 1)
    )

    # Simple statistical flagging: request volume or threat ratio
    # far above the norm for this file -> suspicious behavior
    vol_mean = grouped["total_records"].mean()
    vol_std = grouped["total_records"].std(ddof=0) or 1
    grouped["volume_zscore"] = (grouped["total_records"] - vol_mean) / vol_std

    grouped["flagged"] = (
        (grouped["threat_ratio"] > 0.3) | (grouped["volume_zscore"] > 2)
    )

    flagged = grouped[grouped["flagged"]].sort_values(
        "threat_ratio", ascending=False
    )

    top_offenders = flagged.head(10).to_dict(orient="records")

    return {
        "enabled": True,
        "reason": None,
        "window": window if time_col is not None else "whole-file (no timestamp)",
        "top_offenders": top_offenders,
        "flagged_count": int(flagged.shape[0]),
    }