#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_churn.py  (with churn_top_reasons)

Usage:
    python3 predict_churn.py input.json

Output file:
    churn_result_YYYYMMDD_HHMMSS.json
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib


DEFAULT_MODEL_PATH = Path("path")
DEFAULT_ART_PATH = Path("path")

DATE_COLS = [
    "First_Policy_Date_Gre",
    "Last_Renewal_Date_Gre",
    "Last_Expiration_Date",
]


# ---------------------------------------------------------
# Feature engineering: same as training (single row)
# ---------------------------------------------------------
def engineer_single_row(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse dates
    for c in DATE_COLS:
        if c not in df.columns:
            raise KeyError(f"Missing required date column: {c}")
        df[c] = pd.to_datetime(df[c], errors="coerce",
                               infer_datetime_format=True)

    ref = df["Last_Expiration_Date"].max()
    ref = pd.Timestamp(ref) if pd.notnull(
        ref) else pd.Timestamp.today().normalize()

    X = pd.DataFrame(index=df.index)

    required_numeric = [
        "Years_With_Company",
        "Total_Premium_Paid",
        "Renewal_Count",
        "Avg_Claim_Amount",
        "Claim_Count",
    ]
    for c in required_numeric:
        if c not in df.columns:
            raise KeyError(f"Missing required numeric column: {c}")

    X["years_with_company"] = df["Years_With_Company"]
    X["total_premium_paid"] = df["Total_Premium_Paid"]
    X["renewal_count"] = df["Renewal_Count"]
    X["avg_claim_amount"] = df["Avg_Claim_Amount"]
    X["claim_count"] = df["Claim_Count"]

    X["days_since_first_policy"] = (ref - df["First_Policy_Date_Gre"]).dt.days
    X["days_since_last_renewal"] = (ref - df["Last_Renewal_Date_Gre"]).dt.days
    X["days_to_expiration"] = (df["Last_Expiration_Date"] - ref).dt.days

    X = X.replace([np.inf, -np.inf], np.nan)
    return X


# ---------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------
def load_churn_artifacts(model_path: Path, art_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not art_path.is_file():
        raise FileNotFoundError(art_path)

    model = joblib.load(model_path)

    with art_path.open("r", encoding="utf-8") as f:
        art = json.load(f)

    feature_order = art["feature_order"]
    medians = art["medians"]

    return model, feature_order, medians


# ---------------------------------------------------------
# Simple, fast top_reasons (proxy-SHAP)
# ---------------------------------------------------------
def compute_top_reasons(X_row: pd.Series, top_k: int = 3):
    """
    Compute top_k reasons using a simple magnitude scoring of feature values.
    This is a stable, SHAP-free version suitable for production.
    """
    # Use absolute value as impact proxy
    scores = X_row.abs().sort_values(ascending=False)

    # Return top feature names
    return list(scores.index[:top_k])


# ---------------------------------------------------------
# Churn prediction logic
# ---------------------------------------------------------
def predict_churn_for_values(values: dict, model, feature_order, medians):
    df = pd.DataFrame([values])

    # 1) Feature engineering
    X = engineer_single_row(df)

    # 2) Align columns
    X = X.reindex(columns=feature_order).astype(float)

    # Fill missing
    for c, med in medians.items():
        X[c] = X[c].fillna(med)
    X = X.fillna(0.0)

    # 3) Predict churn probability
    p = float(model.predict_proba(X)[0, 1])

    # 4) Determine churn segment
    if p >= 0.7:
        seg = "High"
    elif p >= 0.4:
        seg = "Mid"
    else:
        seg = "Low"

    # 5) Compute top reasons
    top_reasons = compute_top_reasons(X.iloc[0], top_k=3)

    return {
        "p_churn": p,
        "churn_segment": seg,
        "churn_top_reasons": top_reasons,
    }


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_json")
    ap.add_argument("--model-path",     default=str(DEFAULT_MODEL_PATH))
    ap.add_argument("--artifacts-path", default=str(DEFAULT_ART_PATH))
    args = ap.parse_args()

    input_path = Path(args.input_json)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "values" not in data:
        raise ValueError("JSON must contain top-level 'values'.")

    values = data["values"]

    # Load model
    model, feature_order, medians = load_churn_artifacts(
        Path(args.model_path),
        Path(args.artifacts_path)
    )

    # Predict
    result = predict_churn_for_values(values, model, feature_order, medians)

    # Make output JSON
    out = {"values": dict(values)}
    out["values"]["p_churn"] = round(result["p_churn"], 6)
    out["values"]["churn_segment"] = result["churn_segment"]
    out["values"]["churn_top_reasons"] = result["churn_top_reasons"]

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = Path(f"churn_result_{ts}.json")

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved churn result to: {out_file}")


if __name__ == "__main__":
    main()


"""
HINT: python3 predict_churn.py churn_test.json
"""
