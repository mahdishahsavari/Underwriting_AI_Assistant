import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")


try:
    from catboost import CatBoostClassifier
except:
    CatBoostClassifier = None
try:
    from lightgbm import LGBMClassifier
except:
    LGBMClassifier = None

LABEL_COL = "Churn_Status"  # target column
DATE_COLS = ["First_Policy_Date_Gre",
             "Last_Renewal_Date_Gre", "Last_Expiration_Date"]
NUM_SAFE_NONNEG = ["Years_With_Company", "Total_Premium_Paid",
                   "Renewal_Count", "Avg_Claim_Amount", "Claim_Count"]


def _parse_dates(df):
    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce",
                               infer_datetime_format=True)
    return df


def _winsorize(s, p=0.995):
    if s.isna().all():
        return s
    hi = s.quantile(p)
    lo = s.quantile(1-p)
    return s.clip(lower=lo, upper=hi)


def _clean_and_engineer(df):
    key_cols = [c for c in ["Customer_ID",
                            "Last_Expiration_Date"] if c in df.columns]
    df = df.drop_duplicates(
        subset=key_cols if key_cols else None, keep="last").copy()
    df = _parse_dates(df)
    for c in NUM_SAFE_NONNEG:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] < 0, c] = np.nan
    for c in ["Total_Premium_Paid", "Avg_Claim_Amount"]:
        if c in df.columns:
            df[c] = _winsorize(df[c].astype(float), p=0.995)

    ref_date = df["Last_Expiration_Date"].max()
    ref_date = pd.Timestamp(ref_date) if pd.notnull(
        ref_date) else pd.Timestamp.today().normalize()

    feat = pd.DataFrame(index=df.index)
    m = {
        "years_with_company": "Years_With_Company",
        "total_premium_paid": "Total_Premium_Paid",
        "renewal_count": "Renewal_Count",
        "avg_claim_amount": "Avg_Claim_Amount",
        "claim_count": "Claim_Count",
    }
    for new, old in m.items():
        feat[new] = df[old].astype(float) if old in df.columns else np.nan

    feat["days_since_first_policy"] = (
        ref_date - df["First_Policy_Date_Gre"]).dt.days
    feat["days_since_last_renewal"] = (
        ref_date - df["Last_Renewal_Date_Gre"]).dt.days
    feat["days_to_expiration"] = (
        df["Last_Expiration_Date"] - ref_date).dt.days

    feat = feat.replace([np.inf, -np.inf], np.nan)
    medians = feat.median(numeric_only=True).to_dict()
    feat = feat.fillna(medians)

    feat["days_since_first_policy"] = feat["days_since_first_policy"].clip(
        -365*5, 365*50)
    feat["days_since_last_renewal"] = feat["days_since_last_renewal"].clip(
        -365*5, 365*5)
    feat["days_to_expiration"] = feat["days_to_expiration"].clip(-365*5, 365*5)

    assert LABEL_COL in df.columns, f"Label '{LABEL_COL}' not found"
    y = df[LABEL_COL].astype(int).values
    return feat, y, medians


def _fit_model(algo, X, y):
    if algo == "catboost":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost not installed")
        m = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC",
                               depth=6, learning_rate=0.08, iterations=800,
                               l2_leaf_reg=3.0, random_seed=42, verbose=False)
        m.fit(X, y)
        return m
    if algo == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm not installed")
        m = LGBMClassifier(objective="binary", boosting_type="gbdt",
                           num_leaves=63, learning_rate=0.05, n_estimators=1200,
                           reg_lambda=1.0, subsample=0.9, colsample_bytree=0.9,
                           random_state=42, n_jobs=-1)
        m.fit(X, y)
        return m
    raise ValueError("algo must be catboost or lightgbm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--algo", choices=["catboost", "lightgbm"], required=True)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--out_dir", default="artifacts")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)
    X, y, med = _clean_and_engineer(df)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42)

    base = _fit_model(args.algo, Xtr, ytr)
    cal = CalibratedClassifierCV(base, method="isotonic", cv=3).fit(Xtr, ytr)

    p = cal.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    ap_ = average_precision_score(yte, p)
    brier = brier_score_loss(yte, p)
    print(f"[{args.algo}] AUC={auc:.4f} | AP={ap_:.4f} | Brier={brier:.5f}")
    print(classification_report(yte, (p >= 0.5).astype(int), digits=3))

    joblib.dump(cal, f"{args.out_dir}/churn_{args.algo}_calibrated.joblib")
    with open(f"{args.out_dir}/churn_preprocess_artifacts.json", "w", encoding="utf-8") as f:
        json.dump({"feature_order": list(X.columns), "medians": med},
                  f, ensure_ascii=False, indent=2)

    if hasattr(base, "feature_importances_"):
        imp = pd.Series(base.feature_importances_,
                        index=X.columns).sort_values(ascending=False)
        imp.to_csv(f"{args.out_dir}/feature_importance_{args.algo}.csv")
        print("Top features:\n", imp.head(10))

    print("DONE")


if __name__ == "__main__":
    main()
