import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/raw/cardio_train.csv"
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

def save_plot(name):
    os.makedirs(FIG_PATH, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, name))
    plt.close()

def main():
    df = pd.read_csv(DATA_PATH, sep=";")
    df = preprocess(df)

    # Target Distribution
    df["cardio"].value_counts().plot(kind="bar")
    plt.title("Target Distribution")
    save_plot("target_distribution.png")

    # Age vs Target
    df[df["cardio"]==0]["age_years"].plot(kind="hist", alpha=0.7, bins=40, label="No Disease")
    df[df["cardio"]==1]["age_years"].plot(kind="hist", alpha=0.7, bins=40, label="Disease")
    plt.legend()
    plt.title("Age Distribution by Target")
    save_plot("age_distribution.png")

    # BMI vs Target
    df[df["cardio"]==0]["bmi"].plot(kind="hist", alpha=0.7, bins=40, label="No Disease")
    df[df["cardio"]==1]["bmi"].plot(kind="hist", alpha=0.7, bins=40, label="Disease")
    plt.legend()
    plt.title("BMI Distribution by Target")
    save_plot("bmi_distribution.png")

    # Correlation Heatmap
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10,8))
    plt.imshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar()
    plt.title("Correlation Heatmap")
    save_plot("correlation_heatmap.png")

    print("✅ EDA complete. Check reports/figures")

if __name__ == "__main__":
    main()