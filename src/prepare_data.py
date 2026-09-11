from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "telco_churn.csv"
OUT = ROOT / "data" / "processed" / "telco_churn_clean.csv"

df = pd.read_csv(RAW)

# Clean numeric billing field
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Tenure=0 customers can have no accumulated charges
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Remove identifier from modeling data
df = df.drop(columns=["customerID"])

# Binary target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Useful business features
df["AvgMonthlySpend"] = np.where(
    df["tenure"] > 0,
    df["TotalCharges"] / df["tenure"],
    df["MonthlyCharges"]
)

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12", "13-24", "25-48", "49-72"]
)

df["HighMonthlyCharge"] = (df["MonthlyCharges"] >= df["MonthlyCharges"].median()).astype(int)

df.to_csv(OUT, index=False)

print("Cleaned dataset:", OUT)
print("Shape:", df.shape)
print("Churn rate:", round(df["Churn"].mean() * 100, 2), "%")
