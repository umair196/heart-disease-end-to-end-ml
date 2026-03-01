import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, RocCurveDisplay

DATA_PATH = "data/raw/cardio_train.csv"
MODEL_PATH = "models/heart_disease_pipeline.joblib"
FIG_PATH = "reports/figures"

def preprocess(df):
    df = df.copy()

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df["age_years"] = df["age"] / 365.25
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
    df = df.drop(columns=["age"])

    df = df[(df["ap_hi"] > 0) & (df["ap_lo"] > 0)]
    df = df[(df["ap_hi"] < 300) & (df["ap_lo"] < 200)]

    return df

def main():
    df = pd.read_csv(DATA_PATH, sep=";")
    df = preprocess(df)

    X = df.drop("cardio", axis=1)
    y = df["cardio"]

    cat_cols = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
    num_cols = [col for col in X.columns if col not in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=2000))
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:,1]

    print("ROC-AUC:", roc_auc_score(y_test, probs))
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    # ROC Curve
    RocCurveDisplay.from_predictions(y_test, probs)
    os.makedirs(FIG_PATH, exist_ok=True)
    plt.savefig(os.path.join(FIG_PATH, "roc_curve.png"))

    print("✅ Model trained and saved.")

if __name__ == "__main__":
    main()