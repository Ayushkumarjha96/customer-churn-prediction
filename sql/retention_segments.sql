-- High-value customers at risk
SELECT
    customerID,
    tenure,
    Contract,
    MonthlyCharges,
    TotalCharges,
    PaymentMethod,
    TechSupport,
    Churn
FROM telco_churn
WHERE Churn = 1
  AND MonthlyCharges >= (
      SELECT AVG(MonthlyCharges) FROM telco_churn
  )
ORDER BY TotalCharges DESC;
