import json, joblib, argparse
import numpy as np, pandas as pd

def load_artifacts(model_path, art_path):
    model = joblib.load(model_path)
    art = json.load(open(art_path, "r", encoding="utf-8"))
    return model, art["feature_order"], art["medians"]

def engineer(df):
    df = df.copy()
    for c in ["First_Policy_Date_Gre","Last_Renewal_Date_Gre","Last_Expiration_Date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
    ref = df["Last_Expiration_Date"].max()
    ref = pd.Timestamp(ref) if pd.notnull(ref) else pd.Timestamp.today().normalize()

    X = pd.DataFrame(index=df.index)
    X["years_with_company"] = df["Years_With_Company"]
    X["total_premium_paid"] = df["Total_Premium_Paid"]
    X["renewal_count"]      = df["Renewal_Count"]
    X["avg_claim_amount"]   = df["Avg_Claim_Amount"]
    X["claim_count"]        = df["Claim_Count"]
    X["days_since_first_policy"] = (ref - df["First_Policy_Date_Gre"]).dt.days
    X["days_since_last_renewal"] = (ref - df["Last_Renewal_Date_Gre"]).dt.days
    X["days_to_expiration"]      = (df["Last_Expiration_Date"] - ref).dt.days

    X = X.replace([np.inf,-np.inf], np.nan)
    return X, ref

def unwrap_tree_model(model):
    # Unwrap scikit's CalibratedClassifierCV to the underlying tree model (LightGBM/CatBoost)
    # Supported patterns across sklearn versions:
    # 1) model.base_estimator
    # 2) model.calibrated_classifiers_[0].estimator
    if model.__class__.__name__ == "CalibratedClassifierCV":
        if hasattr(model, "base_estimator") and model.base_estimator is not None:
            return model.base_estimator
        if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
            cc = model.calibrated_classifiers_[0]
            if hasattr(cc, "estimator"):
                return cc.estimator
    return model  # already a native tree model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_in", required=True)
    ap.add_argument("--csv_out", default="artifacts/churn_scored.csv")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--art_path", default="artifacts/churn_preprocess_artifacts.json")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--with_shap", action="store_true")
    ap.add_argument("--shap_sample", type=int, default=0, help="0=all rows, otherwise sample N rows for SHAP")
    args = ap.parse_args()

    model, feats, med = load_artifacts(args.model_path, args.art_path)
    df = pd.read_csv(args.csv_in)
    X, ref = engineer(df)
    X = X[feats].fillna(med)

    probs = model.predict_proba(X)[:,1]
    seg = np.where(probs>=0.7,"High", np.where(probs>=0.4,"Mid","Low"))

    out = df.copy()
    out["p_churn"] = probs
    out["segment"] = seg

    if args.with_shap:
        import shap

        base = unwrap_tree_model(model)
        # Safety: if SHAP still doesn't support the unwrapped model, skip gracefully
        try:
            explainer = shap.TreeExplainer(base)
        except Exception as e:
            print(f"[WARN] SHAP TreeExplainer unsupported for model={type(base)}; skipping SHAP. Error: {e}")
            out["top_reasons"] = ""
            out.to_csv(args.csv_out, index=False)
            print("Saved:", args.csv_out)
            return

        if args.shap_sample and args.shap_sample < len(X):
            samp_idx = np.random.RandomState(42).choice(len(X), size=args.shap_sample, replace=False)
            X_shap = X.iloc[samp_idx]
            index_target = X.index[samp_idx]
        else:
            X_shap = X
            index_target = X.index

        sv = explainer.shap_values(X_shap)
        if isinstance(sv, list):  # e.g., [neg_class, pos_class]
            sv = sv[1]
        vals = np.abs(sv)

        top_names = []
        for i in range(X_shap.shape[0]):
            idx = np.argsort(-vals[i])[:args.topk]
            top_names.append(",".join([X_shap.columns[j] for j in idx]))

        out["top_reasons"] = ""
        out.loc[index_target, "top_reasons"] = top_names
    else:
        out["top_reasons"] = ""

    out.to_csv(args.csv_out, index=False)
    print("Saved:", args.csv_out)

if __name__=="__main__":
    main()
