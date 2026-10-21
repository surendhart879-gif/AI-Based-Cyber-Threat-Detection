"""
train_model.py

Run this ONCE with your labeled training CSV to generate the files
app.py needs:
    model/threat_model.pkl
    model/feature_columns.pkl
    model/model_meta.json

Usage:
    python train_model.py path/to/your_training_data.csv
    python train_model.py "uploads/uploads/*.csv"   (combine ALL matching files)
"""

import glob
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from preprocess import _find_label_column

MODEL_DIR = "model"


def load_csvs(path_or_pattern):
    """Accepts a single CSV path, OR a glob pattern matching several
    (e.g. 'uploads/uploads/*.csv'), and returns one combined DataFrame.
    This matters a lot for CICIDS2017-style data, where each day's
    file may contain only one traffic type (e.g. Monday = all
    BENIGN) — training on a single day teaches the model only one
    class."""
    matches = glob.glob(path_or_pattern)
    if not matches:
        # Not a glob pattern / no matches -> treat as a literal path
        matches = [path_or_pattern]

    frames = []
    for f in matches:
        print(f"Loading {f} ...")
        frames.append(pd.read_csv(f))

    if len(frames) > 1:
        print(f"Combined {len(frames)} files into one training set.")
        return pd.concat(frames, ignore_index=True)
    return frames[0]


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_model.py path/to/training_data.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    data = load_csvs(csv_path)
    data.columns = [c.strip() for c in data.columns]

    label_col = _find_label_column(data)
    if label_col is None:
        print(
            "Could not find a label column. Expected one of: "
            "Label, Class, Attack, attack_type (case-insensitive). "
            "Rename your target column and re-run."
        )
        sys.exit(1)

    y = data[label_col]
    X = data.drop(columns=[label_col])

    # Keep only numeric columns as features
    X = X.select_dtypes(include=["number"])
    # CICIDS2017-style data commonly has inf/-inf values (from
    # divide-by-zero rate calculations like Flow Bytes/s) plus some
    # NaNs. Replace both with 0 so the model can train.
    X = X.mask(np.isinf(X), np.nan).fillna(0)
    feature_columns = list(X.columns)

    if len(feature_columns) == 0:
        print("No numeric feature columns found in the CSV.")
        sys.exit(1)

    print(f"Using {len(feature_columns)} feature columns: {feature_columns}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    print("Training RandomForestClassifier ...")
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {acc:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "threat_model.pkl"))
    joblib.dump(feature_columns, os.path.join(MODEL_DIR, "feature_columns.pkl"))

    meta = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 200,
        "accuracy": round(float(acc), 4),
        "test_accuracy": round(float(acc), 4),
        "samples": len(X_train),
        "features": len(feature_columns),
        "num_features": len(feature_columns),
        "classes": sorted(y.unique().tolist(), key=str),
        "trained_on": os.path.basename(csv_path),
    }
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {MODEL_DIR}/ :")
    print("  threat_model.pkl")
    print("  feature_columns.pkl")
    print("  model_meta.json")


if __name__ == "__main__":
    main()