from pathlib import Path
import requests

URL = "https://raw.githubusercontent.com/IBM/watsonx-ai-samples/master/cpd4.5/data/customer_churn/WA_FnUseC_TelcoCustomerChurn.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco_churn.csv"

OUT.parent.mkdir(parents=True, exist_ok=True)

if OUT.exists():
    print(f"Dataset already exists: {OUT}")
else:
    print("Downloading IBM Telco Customer Churn dataset...")
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    OUT.write_bytes(response.content)
    print(f"Saved: {OUT}")
