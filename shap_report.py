import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

RAW_PATH = "data/raw/cardio_train.csv"
MODEL_PATH = "models/best_model.joblib"
OUT_DIR = "reports/figures"

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df["age_years"] = df["age"] / 365.25
    df = df.drop(columns=["age"])
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
    df = df[(df["ap_hi"] > 0) & (df["ap_lo"] > 0)]
    df = df[(df["ap_hi"] < 300) & (df["ap_lo"] < 200)]
    return df

def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError("Run: python src\\download_data.py")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Train best model first: python src\\train_compare.py")

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(RAW_PATH, sep=";")
    df = prepare(df)

    X = df.drop(columns=["cardio"])
    y = df["cardio"].astype(int)

    pipe = joblib.load(MODEL_PATH)

    # Split pipeline parts
    preprocess = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    # Transform X into model-ready matrix
    X_trans = preprocess.transform(X)

    # Get feature names after preprocessing
    try:
        feature_names = preprocess.get_feature_names_out()
    except Exception:
        feature_names = [f"f_{i}" for i in range(X_trans.shape[1])]

    # Sample for SHAP speed
    n = min(5000, X_trans.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X_trans.shape[0], size=n, replace=False)
    X_s = X_trans[idx]

    # SHAP explainer: tree models vs linear vs fallback
    try:
        # Works well for XGBoost / RandomForest
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_s)
    except Exception:
        # Generic fallback
        explainer = shap.Explainer(model, X_s)
        shap_values = explainer(X_s)

    # Summary plot (bar)
    plt.figure()
    shap.summary_plot(shap_values, X_s, feature_names=feature_names, plot_type="bar", show=False)
    out1 = os.path.join(OUT_DIR, "shap_summary_bar.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=150)
    plt.close()
    print("✅ Saved:", out1)

    # Summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, X_s, feature_names=feature_names, show=False)
    out2 = os.path.join(OUT_DIR, "shap_summary_beeswarm.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=150)
    plt.close()
    print("✅ Saved:", out2)

if __name__ == "__main__":
    main()