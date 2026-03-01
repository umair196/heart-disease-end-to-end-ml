import joblib
import pandas as pd

MODEL_PATH = "models/heart_disease_pipeline.joblib"

def main():
    model = joblib.load(MODEL_PATH)

    sample = {
        "gender": 2,
        "height": 170,
        "weight": 80,
        "ap_hi": 140,
        "ap_lo": 90,
        "cholesterol": 2,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1,
        "age_years": 50,
        "bmi": 80 / ((170/100)**2)
    }

    X = pd.DataFrame([sample])
    prob = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]

    print("Disease Probability:", round(prob, 4))
    print("Prediction (0=no, 1=yes):", pred)

if __name__ == "__main__":
    main()