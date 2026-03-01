import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, RocCurveDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


RAW_PATH = "data/raw/cardio_train.csv"
FIG_DIR = "reports/figures"
MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
RESULTS_PATH = "reports/model_results.json"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # age days -> years
    df["age_years"] = df["age"] / 365.25
    df = df.drop(columns=["age"])

    # BMI
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

    # simple sanity filters
    df = df[(df["ap_hi"] > 0) & (df["ap_lo"] > 0)]
    df = df[(df["ap_hi"] < 300) & (df["ap_lo"] < 200)]
    return df


def savefig(name: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"✅ Saved: {path}")


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    cat_cols = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return preprocess, num_cols, cat_cols


def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Dataset not found at {RAW_PATH}. Run: python src\\download_data.py")

    df = pd.read_csv(RAW_PATH, sep=";")
    df = prepare(df)

    target = "cardio"
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    preprocess, _, _ = build_preprocessor(X)

    # Holdout split for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Models to compare
    models = {
        "LogisticRegression": LogisticRegression(max_iter=3000),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    best_name = None
    best_cv_auc = -1.0
    best_pipeline = None

    print("\n===== 5-Fold CV (ROC-AUC) Comparison =====")
    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
        # CV AUC
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        mean_auc = float(np.mean(cv_auc))
        std_auc = float(np.std(cv_auc))
        results[name] = {"cv_auc_mean": mean_auc, "cv_auc_std": std_auc}

        print(f"{name:18s}  AUC = {mean_auc:.4f} ± {std_auc:.4f}")

        if mean_auc > best_cv_auc:
            best_cv_auc = mean_auc
            best_name = name
            best_pipeline = pipe

    print(f"\n✅ Best model by CV AUC: {best_name} (AUC={best_cv_auc:.4f})")

    # Fit best on full training set and evaluate on holdout test
    best_pipeline.fit(X_train, y_train)
    proba = best_pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    test_auc = float(roc_auc_score(y_test, proba))
    test_acc = float(accuracy_score(y_test, pred))
    results[best_name]["test_auc"] = test_auc
    results[best_name]["test_accuracy"] = test_acc

    print("\n===== Holdout Test Metrics (Best Model) =====")
    print("Model:", best_name)
    print("ROC-AUC:", round(test_auc, 4))
    print("Accuracy:", round(test_acc, 4))
    print("\nClassification Report:\n", classification_report(y_test, pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

    # ROC curve plot
    plt.figure()
    RocCurveDisplay.from_predictions(y_test, proba)
    plt.title(f"ROC Curve ({best_name})")
    savefig("best_model_roc_curve.png")
    plt.close()

    # Save best model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    print(f"✅ Saved best model to: {BEST_MODEL_PATH}")

    # Save results json
    os.makedirs("reports", exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()