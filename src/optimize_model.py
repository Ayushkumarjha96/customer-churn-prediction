from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# =====================================================
# PATHS
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "telco_churn_clean.csv"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)


# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Churn"])
y = df["Churn"]


# =====================================================
# IDENTIFY FEATURES
# =====================================================

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()


# =====================================================
# PREPROCESSING
# =====================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        ),

        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]),
            categorical_features
        )
    ]
)


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# =====================================================
# MODELS
# =====================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42
    )
}


# =====================================================
# CROSS VALIDATION
# =====================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


results = {}


print("\n")
print("=" * 70)
print("MODEL CROSS-VALIDATION")
print("=" * 70)


for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    results[name] = {
        "cv_roc_auc_mean": float(scores.mean()),
        "cv_roc_auc_std": float(scores.std())
    }

    print(
        f"{name}: "
        f"{scores.mean():.4f} "
        f"(± {scores.std():.4f})"
    )


# =====================================================
# TRAIN MODELS
# =====================================================

print("\n")
print("=" * 70)
print("TEST SET PERFORMANCE")
print("=" * 70)


test_results = {}


for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    test_results[name] = {

        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),

        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_test,
                probabilities
            )
        )
    }

    print(f"\n{name}")

    for metric, value in test_results[name].items():

        print(
            f"{metric:10s}: {value:.4f}"
        )


# =====================================================
# SELECT BEST MODEL
# =====================================================

best_model_name = max(
    test_results,
    key=lambda x: test_results[x]["roc_auc"]
)

print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print(best_model_name)


# =====================================================
# FIT BEST MODEL
# =====================================================

best_model = models[best_model_name]

best_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", best_model)
])

best_pipeline.fit(
    X_train,
    y_train
)


# =====================================================
# THRESHOLD OPTIMIZATION
# =====================================================

probabilities = best_pipeline.predict_proba(
    X_test
)[:, 1]


threshold_results = []


for threshold in np.arange(
    0.20,
    0.81,
    0.01
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    threshold_results.append({

        "threshold": threshold,

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    })


threshold_df = pd.DataFrame(
    threshold_results
)


best_threshold_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

best_threshold = float(
    best_threshold_row["threshold"]
)


print("\n")
print("=" * 70)
print("THRESHOLD OPTIMIZATION")
print("=" * 70)

print(
    f"Best threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Precision: "
    f"{best_threshold_row['precision']:.4f}"
)

print(
    f"Recall: "
    f"{best_threshold_row['recall']:.4f}"
)

print(
    f"F1 Score: "
    f"{best_threshold_row['f1']:.4f}"
)


# =====================================================
# SAVE FINAL MODEL
# =====================================================

joblib.dump(
    best_pipeline,
    MODEL_DIR / "final_churn_model.joblib"
)


# =====================================================
# SAVE CONFIGURATION
# =====================================================

configuration = {

    "best_model": best_model_name,

    "threshold": best_threshold,

    "cross_validation": results,

    "test_results": test_results
}


with open(
    MODEL_DIR / "final_model_config.json",
    "w"
) as file:

    json.dump(
        configuration,
        file,
        indent=4
    )


print("\n")
print("=" * 70)
print("FINAL MODEL SAVED")
print("=" * 70)

print(
    MODEL_DIR / "final_churn_model.joblib"
)