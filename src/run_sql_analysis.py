from pathlib import Path
import sqlite3
import pandas as pd

# Project root
ROOT = Path(__file__).resolve().parents[1]

# Database location
DB_PATH = ROOT / "data" / "churn.db"

# Connect to database
connection = sqlite3.connect(DB_PATH)


def run_query(title, query):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    result = pd.read_sql_query(query, connection)

    print(result.to_string(index=False))


# ==========================================
# 1. OVERALL CHURN
# ==========================================

run_query(
    "OVERALL CHURN",
    """
    SELECT
        COUNT(*) AS total_customers,
        SUM(Churn) AS churned_customers,
        ROUND(AVG(Churn) * 100, 2) AS churn_rate
    FROM telco_churn;
    """
)


# ==========================================
# 2. CHURN BY CONTRACT
# ==========================================

run_query(
    "CHURN BY CONTRACT",
    """
    SELECT
        Contract,
        COUNT(*) AS customers,
        ROUND(AVG(Churn) * 100, 2) AS churn_rate
    FROM telco_churn
    GROUP BY Contract
    ORDER BY churn_rate DESC;
    """
)


# ==========================================
# 3. CHURN BY PAYMENT METHOD
# ==========================================

run_query(
    "CHURN BY PAYMENT METHOD",
    """
    SELECT
        PaymentMethod,
        COUNT(*) AS customers,
        ROUND(AVG(Churn) * 100, 2) AS churn_rate
    FROM telco_churn
    GROUP BY PaymentMethod
    ORDER BY churn_rate DESC;
    """
)


# ==========================================
# 4. CHURN BY TECHNICAL SUPPORT
# ==========================================

run_query(
    "CHURN BY TECHNICAL SUPPORT",
    """
    SELECT
        TechSupport,
        COUNT(*) AS customers,
        ROUND(AVG(Churn) * 100, 2) AS churn_rate
    FROM telco_churn
    GROUP BY TechSupport
    ORDER BY churn_rate DESC;
    """
)


# ==========================================
# 5. CHURN BY INTERNET SERVICE
# ==========================================

run_query(
    "CHURN BY INTERNET SERVICE",
    """
    SELECT
        InternetService,
        COUNT(*) AS customers,
        ROUND(AVG(Churn) * 100, 2) AS churn_rate
    FROM telco_churn
    GROUP BY InternetService
    ORDER BY churn_rate DESC;
    """
)


# ==========================================
# 6. CHURN BY TENURE GROUP
# ==========================================

run_query(
    "CHURN BY TENURE GROUP",
    """
    SELECT
        TenureGroup,
        COUNT(*) AS customers,
        ROUND(AVG(Churn) * 100, 2) AS churn_rate
    FROM telco_churn
    GROUP BY TenureGroup
    ORDER BY churn_rate DESC;
    """
)


# ==========================================
# 7. AVERAGE MONTHLY CHARGES
# ==========================================

run_query(
    "AVERAGE MONTHLY CHARGES BY CHURN",
    """
    SELECT
        Churn,
        ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges,
        ROUND(AVG(TotalCharges), 2) AS avg_total_charges
    FROM telco_churn
    GROUP BY Churn;
    """
)


# ==========================================
# 8. HIGH VALUE CHURNED CUSTOMERS
# ==========================================

run_query(
    "HIGH VALUE CHURNED CUSTOMERS",
    """
    SELECT
        tenure,
        Contract,
        MonthlyCharges,
        TotalCharges,
        PaymentMethod,
        TechSupport
    FROM telco_churn
    WHERE Churn = 1
    ORDER BY TotalCharges DESC
    LIMIT 20;
    """
)


# ==========================================
# CLOSE DATABASE
# ==========================================

connection.close()

print("\n" + "=" * 60)
print("SQL ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 60)