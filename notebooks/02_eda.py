from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# PATHS
# =========================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "telco_churn_clean.csv"
IMAGE_DIR = ROOT / "images"

IMAGE_DIR.mkdir(exist_ok=True)


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)
print("\nChurn Distribution:")
print(df["Churn"].value_counts())


# =========================
# 1. OVERALL CHURN
# =========================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Churn"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    IMAGE_DIR / "01_churn_distribution.png",
    dpi=300
)

plt.close()


# =========================
# 2. CHURN BY CONTRACT
# =========================

contract_churn = (
    df.groupby("Contract")["Churn"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

plt.figure(figsize=(8, 5))

contract_churn.plot(kind="bar")

plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    IMAGE_DIR / "02_churn_by_contract.png",
    dpi=300
)

plt.close()


# =========================
# 3. CHURN BY TENURE
# =========================

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="tenure"
)

plt.title("Tenure Distribution by Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Tenure (Months)")

plt.tight_layout()

plt.savefig(
    IMAGE_DIR / "03_churn_by_tenure.png",
    dpi=300
)

plt.close()


# =========================
# 4. MONTHLY CHARGES
# =========================

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges by Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Monthly Charges")

plt.tight_layout()

plt.savefig(
    IMAGE_DIR / "04_monthly_charges.png",
    dpi=300
)

plt.close()


# =========================
# 5. PAYMENT METHOD
# =========================

payment_churn = (
    df.groupby("PaymentMethod")["Churn"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

plt.figure(figsize=(10, 6))

payment_churn.plot(kind="bar")

plt.title("Churn Rate by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=30, ha="right")

plt.tight_layout()

plt.savefig(
    IMAGE_DIR / "05_churn_by_payment.png",
    dpi=300
)

plt.close()


# =========================
# 6. INTERNET SERVICE
# =========================

internet_churn = (
    df.groupby("InternetService")["Churn"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

plt.figure(figsize=(8, 5))

internet_churn.plot(kind="bar")

plt.title("Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    IMAGE_DIR / "06_churn_by_internet.png",
    dpi=300
)

plt.close()


# =========================
# 7. TECH SUPPORT
# =========================

support_churn = (
    df.groupby("TechSupport")["Churn"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

plt.figure(figsize=(9, 5))

support_churn.plot(kind="bar")

plt.title("Churn Rate by Technical Support")
plt.xlabel("Technical Support")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    IMAGE_DIR / "07_churn_by_support.png",
    dpi=300
)

plt.close()


# =========================
# 8. SENIOR CITIZEN
# =========================

senior_churn = (
    df.groupby("SeniorCitizen")["Churn"]
    .mean()
    * 100
)

plt.figure(figsize=(7, 5))

senior_churn.plot(kind="bar")

plt.title("Churn Rate by Senior Citizen Status")
plt.xlabel("Senior Citizen (0 = No, 1 = Yes)")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    IMAGE_DIR / "08_churn_by_senior.png",
    dpi=300
)

plt.close()


# =========================
# PRINT BUSINESS INSIGHTS
# =========================

print("\n" + "=" * 50)
print("BUSINESS INSIGHTS")
print("=" * 50)

print("\nChurn by Contract:")
print(contract_churn.round(2))

print("\nChurn by Payment Method:")
print(payment_churn.round(2))

print("\nChurn by Internet Service:")
print(internet_churn.round(2))

print("\nChurn by Technical Support:")
print(support_churn.round(2))

print("\nEDA completed successfully!")
print(f"Charts saved to: {IMAGE_DIR}")