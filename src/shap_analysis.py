from pathlib import Path

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt


# ==========================================
# PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "telco_churn_clean.csv"
MODEL_PATH = ROOT / "models" / "final_churn_model.joblib"
IMAGE_DIR = ROOT / "images"

IMAGE_DIR.mkdir(exist_ok=True)


# ==========================================
# LOAD DATA AND MODEL
# ==========================================

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)

X = df.drop(columns=["Churn"])


# ==========================================
# TRANSFORM DATA
# ==========================================

preprocessor = model.named_steps["preprocessor"]

classifier = model.named_steps["model"]

X_transformed = preprocessor.transform(X)


# ==========================================
# GET FEATURE NAMES
# ==========================================

feature_names = preprocessor.get_feature_names_out()


# ==========================================
# SHAP EXPLAINER
# ==========================================

explainer = shap.Explainer(
    classifier,
    X_transformed,
    feature_names=feature_names
)

shap_values = explainer(
    X_transformed
)


# ==========================================
# GLOBAL FEATURE IMPORTANCE
# ==========================================

plt.figure()

shap.plots.bar(
    shap_values,
    max_display=15,
    show=False
)

plt.title("Top Factors Influencing Customer Churn")

plt.tight_layout()

plt.savefig(
    IMAGE_DIR / "shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# SHAP SUMMARY
# ==========================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    IMAGE_DIR / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("=" * 60)
print("SHAP ANALYSIS COMPLETED")
print("=" * 60)

print("\nGenerated files:")

print(
    IMAGE_DIR / "shap_feature_importance.png"
)

print(
    IMAGE_DIR / "shap_summary.png"
)