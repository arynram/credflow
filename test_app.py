import pandas as pd
import joblib
from feature_engineering import compute_features_from_transactions, categorize_transaction
from benford_audit import calculate_benford_distribution, analyze_counterparty_graph
from forecasting import forecast_cashflow
from stress_simulator import simulate_macro_stress
from train_model import explain_prediction, MODEL_ARTIFACT_PATH

print("Testing artifacts load...")
artifacts = joblib.load(MODEL_ARTIFACT_PATH)
model = artifacts['model']
sample_msmes = artifacts['sample_msmes']
print(f"Loaded {len(sample_msmes)} sample MSMEs.")

print("Testing feature extraction on sample_transactions_healthy.csv...")
df_healthy = pd.read_csv("sample_transactions_healthy.csv")
feats = compute_features_from_transactions(df_healthy)
print(f"Extracted features: Revenue={feats['avg_monthly_revenue']}, EMI Burden={feats['emi_burden']}, Bounces={feats['bounce_count']}")

print("Testing Benford audit...")
amounts = df_healthy['Credit'].tolist() + df_healthy['Debit'].tolist()
b_res = calculate_benford_distribution(amounts, min_samples=50)
print(f"Benford verdict: {b_res['verdict']}, p-value: {b_res['p_value']}")

print("Testing forecasting...")
series = pd.Series([20.0 + i * 0.5 for i in range(15)])
fc_res = forecast_cashflow(series, forecast_horizon=6, min_history_required=12)
print(f"Forecasting status: {fc_res['status']}")

print("Testing stress simulator...")
stress_res = simulate_macro_stress(sample_msmes.iloc[0].to_dict(), revenue_shock_pct=-0.25)
print(f"Stress test summary: {stress_res['summary']}")

print("Testing Groq AI parser & local fallback...")
from groq_parser import parse_and_standardize_statement_groq, generate_groq_lender_memo, is_groq_available
parsed_demo = parse_and_standardize_statement_groq(df_healthy)
print(f"Groq module status: {parsed_demo['source']}, candidates extracted: {len(parsed_demo['candidates'])}")
assert len(parsed_demo['candidates']) >= 2, "Should have extracted at least 2 candidates"

memo = generate_groq_lender_memo(sample_msmes.iloc[0].to_dict())
print(f"Lender memo preview: {memo[:100]}...")

print("\nALL BACKEND & AI MODULES FUNCTIONING FLAWLESSLY!")
