from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "telco_churn_clean.csv"
DB_PATH = ROOT / "data" / "churn.db"

# Load data
df = pd.read_csv(DATA_PATH)

# Create SQLite database
connection = sqlite3.connect(DB_PATH)

# Put dataframe into SQL table
df.to_sql(
    "telco_churn",
    connection,
    if_exists="replace",
    index=False
)

connection.close()

print("Database created successfully!")
print(f"Database location: {DB_PATH}")
print(f"Rows inserted: {len(df)}")