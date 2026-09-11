from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "telco_churn_clean.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

X = df.drop(columns=["Churn"])
y = df["Churn"]

# Treat engineered category as categorical
categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric = X.select_dtypes(include=["number"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "logistic_regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=400, class_weight="balanced",
        random_state=42, n_jobs=-1
    ),
    "xgboost": XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42
    )
}

results = {}

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)

    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.50).astype(int)

    results[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob)
    }

    print("\n", name)
    print(classification_report(y_test, pred, zero_division=0))

# Select by ROC-AUC for a first-pass baseline
best_name = max(results, key=lambda k: results[k]["roc_auc"])
best_model = models[best_name]

best_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", best_model)
])
best_pipeline.fit(X_train, y_train)

joblib.dump(best_pipeline, MODEL_DIR / "churn_model.joblib")

with open(MODEL_DIR / "model_metrics.json", "w") as f:
    json.dump(
        {"best_model": best_name, "models": results},
        f, indent=2
    )

# Save test predictions for analysis
predictions = X_test.copy()
predictions["ActualChurn"] = y_test.values
predictions["ChurnProbability"] = best_pipeline.predict_proba(X_test)[:, 1]
predictions["RiskLevel"] = pd.cut(
    predictions["ChurnProbability"],
    bins=[-0.01, 0.40, 0.70, 1.01],
    labels=["Low", "Medium", "High"]
)
predictions.to_csv(ROOT / "data" / "processed" / "churn_predictions.csv", index=False)

print(f"\nBest model: {best_name}")
print(json.dumps(results[best_name], indent=2))
