# ChurnIQ — Customer Churn Prediction & Retention Intelligence

> An end-to-end machine learning system that predicts customer churn risk, explains the key drivers behind each prediction, and recommends targeted retention actions.

🔗 **Live Demo:** https://customer-churn-prediction-uwqzvkgagdpwe5gqqqgpan.streamlit.app/
---

## 📌 Project Overview

Customer churn is a major business problem where identifying high-risk customers early can help organizations take proactive retention actions.

**ChurnIQ** is an end-to-end customer churn prediction system built using machine learning, explainable AI, SQL analytics, and Streamlit.

The system allows a user to:

- Enter customer information
- Predict the probability of churn
- Classify the customer into a risk category
- Understand the strongest churn drivers
- Generate targeted retention recommendations
- Analyze customer risk using SQL

---

## 🎯 Key Features

### 🤖 Churn Prediction
Predicts the probability that a customer will leave the service.

### 📊 Risk Classification
Customers are classified into:

- 🟢 Low Risk
- 🟡 Medium Risk
- 🔴 High Risk

### 🔍 Explainable AI

SHAP is used to understand which features contribute most strongly to churn predictions.

### 💡 Retention Intelligence

The system generates targeted retention actions based on customer characteristics and predicted risk.

### 🗄️ SQL Business Analytics

SQL queries are used to analyze:

- Customer churn KPIs
- High-risk customers
- Retention segments
- Customer behavior patterns

### 🌐 Interactive Deployment

The complete ML system is deployed as an interactive Streamlit application.

---

## 🏗️ System Architecture

```text
Customer Data
      ↓
Data Cleaning & Preprocessing
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Machine Learning Model
      ↓
Model Evaluation
      ↓
Threshold Optimization
      ↓
SHAP Explainability
      ↓
Retention Strategy
      ↓
Streamlit Deployment
```

---

## 📊 Model Performance

The initial model evaluation achieved:

| Metric | Score |
|---|---:|
| Accuracy | 73.39% |
| Precision | 49.92% |
| Recall | 79.77% |
| F1 Score | 61.38% |
| ROC-AUC | 84.17% |

The classification threshold was subsequently optimized to **0.55** to provide a better balance between precision and recall for the business use case.

### Optimized Model

| Metric | Score |
|---|---:|
| Threshold | 0.55 |
| Precision | 53.90% |
| Recall | 75.67% |
| F1 Score | 62.96% |

The model prioritizes identifying customers who are likely to churn, making recall an important metric for the retention use case.

---

## 🔎 Explainable AI with SHAP

SHAP (SHapley Additive exPlanations) is used to interpret the machine learning model.

The project generates:

```text
images/shap_feature_importance.png
images/shap_summary.png
```

These visualizations help identify the features that have the greatest influence on churn predictions.

---

## 📈 Exploratory Data Analysis

EDA was performed to identify customer behavior and churn patterns.

Example insight:

Customers without technical support showed a substantially different churn distribution compared with customers receiving technical support.

The project generates multiple visualizations inside:

```text
images/
```

---

## 🗄️ SQL Analytics

The project uses SQLite for business-oriented customer analysis.

SQL analysis includes:

```text
sql/
├── churn_kpis.sql
├── customer_risk.sql
└── retention_segments.sql
```

These queries provide additional business insights beyond the machine learning prediction.

---

## 🛠️ Tech Stack

### Programming
- Python

### Machine Learning
- Scikit-learn
- XGBoost
- Joblib

### Data Analysis
- Pandas
- NumPy

### Visualization
- Matplotlib
- Seaborn

### Explainable AI
- SHAP

### Database
- SQLite
- SQL

### Deployment
- Streamlit Community Cloud

### Development
- VS Code
- Git
- GitHub

---

## 📁 Project Structure

```text
customer-churn-prediction/
│
├── app/
│   └── app.py
│
├── data/
│   └── churn.csv
│
├── images/
│   ├── shap_feature_importance.png
│   ├── shap_summary.png
│   └── ...
│
├── models/
│   ├── final_churn_model.joblib
│   ├── final_model_config.json
│   └── model_metrics.json
│
├── notebooks/
│   ├── 01_project_pipeline.md
│   └── 02_eda.py
│
├── reports/
│
├── sql/
│   ├── churn_kpis.sql
│   ├── customer_risk.sql
│   └── retention_segments.sql
│
├── src/
│   ├── download_data.py
│   ├── eda.py
│   ├── optimize_model.py
│   ├── predict.py
│   ├── prepare_data.py
│   ├── run_sql_analysis.py
│   ├── setup_database.py
│   ├── shap_analysis.py
│   └── train_model.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🚀 Running Locally

Clone the repository:

```bash
git clone https://github.com/Ayushkumarjha96/customer-churn-prediction.git
```

Move into the project:

```bash
cd customer-churn-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app/app.py
```

---

## 🌐 Live Application

The deployed application allows users to interact with the trained model through a web interface.

**Live Demo:**

https://customer-churn-prediction-uwqzvkgagdpwe5gqqqgpan.streamlit.app/

---

## 💼 Business Impact

ChurnIQ demonstrates how machine learning can be transformed into a practical business decision-support system.

Instead of only predicting:

> "Will this customer churn?"

the system answers:

> "How likely is this customer to churn, why are they at risk, and what retention action should be considered?"

This connects machine learning predictions with actionable customer retention strategies.

---

## 🔮 Future Improvements

- Real-time customer data integration
- Model monitoring and drift detection
- Automated retention campaign integration
- Customer lifetime value prediction
- A/B testing of retention strategies
- Advanced model comparison and hyperparameter tuning
- Cloud database integration

---

## 👨‍💻 Author

**Ayush Kumar Jha**

Data Science & Machine Learning Project

GitHub:  
https://github.com/Ayushkumarjha96