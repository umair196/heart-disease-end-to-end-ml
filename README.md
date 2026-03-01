# ❤️ Heart (Cardiovascular) Disease Prediction — End-to-End ML Project

## 📌 Overview
This project is a complete **end-to-end machine learning system** for predicting the risk of cardiovascular (heart) disease based on patient health indicators.

It covers the **full ML lifecycle**:
- Data ingestion & preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model comparison with cross-validation
- Model explainability using SHAP
- Deployment using Streamlit
- Reproducible project structure

The project is designed to demonstrate **industry-level data science practices**.

---

## 🎯 Problem Statement
Cardiovascular disease is one of the leading causes of death worldwide.  
The goal of this project is to build a machine learning model that predicts whether a patient is at risk of heart disease using clinical and lifestyle features, enabling early intervention and better decision-making.

---

## 📊 Dataset
- **Name:** Cardiovascular Disease Dataset  
- **Size:** ~70,000 patient records  
- **Target Variable:** `cardio` (0 = No disease, 1 = Disease)

### Key Features
- Age (converted from days to years)
- Blood pressure (systolic & diastolic)
- Cholesterol & glucose levels
- Height, weight, BMI
- Lifestyle indicators (smoking, alcohol, physical activity)

> The dataset is automatically downloaded for reproducibility and is not stored in the repository.

---

## 🧱 Project Structure
heart-disease-e2e/
│
├── app/
│ └── app.py # Streamlit web application
│
├── src/
│ ├── download_data.py # Auto-download dataset
│ ├── eda.py # EDA & visualizations
│ ├── train_compare.py # Model comparison + CV
│ ├── shap_report.py # SHAP explainability
│
├── models/
│ └── best_model.joblib # Trained ML pipeline
│
├── reports/
│ └── figures/ # EDA, ROC, SHAP plots
│
├── requirements.txt
├── README.md
└── .gitignore


---

## 🔍 Exploratory Data Analysis (EDA)
EDA includes:
- Target class distribution
- Age & BMI distribution by disease status
- Blood pressure relationships
- Correlation heatmap

All plots are saved to:

reports/figures/


---

## 🤖 Model Training & Comparison
Three models were evaluated using **5-fold Stratified Cross-Validation**:

- Logistic Regression (baseline)
- Random Forest
- XGBoost

### Model Selection Criteria
- Primary metric: **ROC-AUC**
- Best-performing model selected based on cross-validation performance

The final model is trained on the full training set and evaluated on a holdout test set.

---

## 📈 Model Explainability (SHAP)
To improve interpretability, **SHAP (SHapley Additive exPlanations)** was used to:
- Identify the most important features
- Understand feature impact on predictions

Generated plots:
- SHAP summary (bar)
- SHAP summary (beeswarm)

Saved in:

reports/figures/


---

## 🌐 Deployment (Streamlit)
An interactive **Streamlit web app** allows users to:
- Enter patient health data
- Get disease probability
- View binary prediction (Disease / No Disease)

### Run locally
```bash
python -m streamlit run app/app.py
Live Demo

👉 Add your Streamlit Cloud URL here after deployment

⚙️ How to Run the Project
1️⃣ Install dependencies
pip install -r requirements.txt
2️⃣ Download dataset
python src/download_data.py
3️⃣ Run EDA
python src/eda.py
4️⃣ Train & compare models
python src/train_compare.py
5️⃣ Generate SHAP explainability
python src/shap_report.py
6️⃣ Launch Streamlit app
python -m streamlit run app/app.py
📊 Results (Update After Training)

Best Model: XGBoost / Random Forest (update)

5-Fold CV ROC-AUC: X.XXXX ± X.XXXX

Holdout ROC-AUC: X.XXXX

Holdout Accuracy: X.XXXX

🧠 Key Learnings

Built a fully reproducible ML pipeline

Applied cross-validation for robust model selection

Used SHAP for transparent and explainable predictions

Deployed an ML model as a production-style web app

🚀 Future Improvements

Hyperparameter tuning (Optuna / GridSearch)

Time-series or longitudinal health data

Model monitoring & drift detection

API-based deployment (FastAPI)

👤 Author

Your Name
Data Scientist | Machine Learning | Python
GitHub: https://github.com/YOUR_USERNAME

LinkedIn: Muhammad Umair