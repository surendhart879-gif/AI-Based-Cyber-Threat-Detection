"""
preprocess.py

Converts a raw uploaded CSV into the numeric feature matrix the
trained model expects, and (if present) extracts the true labels
for accuracy comparison.
"""

import numpy as np
import pandas as pd

# If your dataset's label column is named something else
# (e.g. "Class", "Attack_type"), change this.
LABEL_CANDIDATES = ["Label", "label", "Class", "class", "Attack", "attack_type"]


def _find_label_column(data: pd.DataFrame):
    for col in LABEL_CANDIDATES:
        if col in data.columns:
            return col
    return None


def prepare_features(data: pd.DataFrame, feature_columns):
    """
    Args:
        data: raw DataFrame read from the uploaded CSV.
        feature_columns: list of column names the model was trained on
                          (loaded from model/feature_columns.pkl).

    Returns:
        (features, labels)
        features: DataFrame with exactly `feature_columns`, in that
                   order, numeric, with missing values filled.
        labels: Series of true labels if a label column was found in
                the CSV, else None.
    """
    df = data.copy()

    # Separate out labels if present, so they don't get treated as
    # a feature.
    label_col = _find_label_column(df)
    labels = df[label_col] if label_col is not None else None
    if label_col is not None:
        df = df.drop(columns=[label_col])

    # Clean up column name whitespace (common issue in these datasets)
    df.columns = [c.strip() for c in df.columns]

    # Keep only the columns the model was trained on, in the right
    # order. Any expected column missing from this CSV is added as 0.
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    features = df[feature_columns].copy()

    # Convert everything to numeric; anything that can't convert
    # becomes NaN, then we fill with 0.
    features = features.apply(pd.to_numeric, errors="coerce")

    # NOTE: we deliberately avoid features.replace([inf, -inf], ...)
    # here. On some pandas versions it triggers an internal bug
    # (IndexError: pop index out of range) when combined with
    # apply(pd.to_numeric) just above. numpy.isinf + mask does the
    # same job without touching that code path.
    features = features.mask(np.isinf(features), np.nan)
    features = features.fillna(0)

    return features, labels