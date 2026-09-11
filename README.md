# 🚀 ChurnIQ — Customer Churn Prediction & Retention Intelligence

> An end-to-end machine learning system that predicts customer churn risk, explains the key drivers behind each prediction, and generates targeted retention recommendations.

## 🌐 Live Demo

👉 **[Launch ChurnIQ](https://customer-churn-prediction-uwqzvkgagd pwe5ggqqqpan.streamlit.app/)**

---

## 🎯 Project Overview

Customer churn is a major business problem where identifying high-risk customers early can help organizations take proactive retention actions.

**ChurnIQ** combines:

- 🤖 Machine Learning
- 🔍 Explainable AI
- 📊 Exploratory Data Analysis
- 🗄️ SQL Analytics
- 🎯 Retention Intelligence
- 🌐 Streamlit Deployment

The system answers three important business questions:

> **How likely is this customer to churn?**

> **Why is the customer at risk?**

> **What retention action should be considered?**

---

## ⭐ Key Features

### 🤖 Churn Prediction

Predicts the probability that a customer will leave the service using a trained machine learning model.

### 📊 Risk Classification

Customers are classified into:

- 🟢 Low Risk
- 🟡 Medium Risk
- 🔴 High Risk

The final model uses an optimized classification threshold of **0.55**.

### 🔍 Explainable AI

SHAP is used to identify the features contributing most strongly to churn predictions.

This makes the model more interpretable instead of treating it as a black box.

### 💡 Retention Intelligence

The system generates targeted retention recommendations based on customer characteristics and predicted churn risk.

Example factors include:

- Contract type
- Monthly charges
- Technical support
- Payment method
- Customer tenure
- Internet service

### 🗄️ SQL Business Analytics

SQLite and SQL queries are used to analyze customer risk and retention patterns.

---

# 📈 Model Performance

The model was optimized with a focus on identifying customers who are likely to churn.

| Metric | Score |
|---|---:|
| ROC-AUC | **0.842** |
| Recall | **0.757** |
| Precision | **0.539** |
| F1 Score | **0.630** |
| Classification Threshold | **0.55** |

### Why Recall Matters

For customer churn, missing a genuinely high-risk customer can be more costly than contacting a customer who ultimately stays.

Therefore, the model was optimized to maintain strong **churn recall** while improving the overall F1 score.

---

# 🔍 Explainable AI with SHAP

The project uses SHAP to understand which features have the greatest influence on churn predictions.

Generated visualizations include:

```text
images/shap_feature_importance.png
images/shap_summary.png