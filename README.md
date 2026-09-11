# Customer Churn Prediction & Retention System

An end-to-end Data Science portfolio project for predicting telecom customer churn and turning predictions into actionable retention recommendations.

## Business Goal
Identify customers at high risk of churn, understand the main drivers behind the prediction, and prioritize retention actions.

## Tech Stack
Python, Pandas, NumPy, SQL, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn, Streamlit.

## Dataset
IBM Telco Customer Churn sample dataset:
- 7,043 customer records
- 21 columns
- Target: `Churn`

The raw CSV is downloaded automatically by `src/download_data.py` from IBM's public GitHub sample-data repository.

## Project Structure
```text
customer-churn-prediction/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── images/
├── models/
├── notebooks/
├── reports/
├── sql/
│   ├── churn_kpis.sql
│   ├── customer_risk.sql
│   └── retention_segments.sql
├── src/
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── train_model.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## Run
```bash
pip install -r requirements.txt
python src/download_data.py
python src/prepare_data.py
python src/train_model.py
streamlit run app/app.py
```

## Important
Model metrics are generated when you run the pipeline. Do not put made-up metrics on your resume.

## Resume-ready project title
**Customer Churn Prediction & Retention Intelligence System**

## What makes this project stronger
- Binary churn prediction
- Business-oriented evaluation
- Risk scoring
- Explainability
- Retention recommendations
- SQL analysis
- Interactive Streamlit app
