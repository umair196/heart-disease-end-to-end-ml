import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/best_model.joblib"

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("❤️ Heart (Cardiovascular) Disease Prediction")
st.write("Enter patient details and predict cardiovascular disease risk.")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

pipe = load_model()

# Inputs
age_years = st.number_input("Age (years)", min_value=1.0, max_value=120.0, value=50.0, step=1.0)
gender = st.selectbox("Gender (dataset code)", options=[1, 2], index=1)  # dataset uses 1/2
height = st.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=170.0, step=1.0)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=80.0, step=1.0)
ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=50.0, max_value=250.0, value=140.0, step=1.0)
ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=30.0, max_value=200.0, value=90.0, step=1.0)

cholesterol = st.selectbox("Cholesterol (1=normal, 2=above, 3=high)", options=[1,2,3], index=1)
gluc = st.selectbox("Glucose (1=normal, 2=above, 3=high)", options=[1,2,3], index=0)
smoke = st.selectbox("Smoke", options=[0,1], index=0)
alco = st.selectbox("Alcohol intake", options=[0,1], index=0)
active = st.selectbox("Physically active", options=[0,1], index=1)

bmi = weight / ((height / 100) ** 2)

row = {
    "gender": gender,
    "height": height,
    "weight": weight,
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "cholesterol": cholesterol,
    "gluc": gluc,
    "smoke": smoke,
    "alco": alco,
    "active": active,
    "age_years": age_years,
    "bmi": bmi,
}

X = pd.DataFrame([row])

if st.button("Predict"):
    proba = float(pipe.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)

    st.subheader("Result")
    st.write(f"**Probability of cardiovascular disease:** {proba:.3f}")
    st.write(f"**Prediction:** {'Disease' if pred==1 else 'No Disease'}")