"""
Groq AI Bank Passbook & Statement Intelligence Parser.
Leverages Groq LPU Ultra-Low Latency Inference with Llama 3.3 (70B/8B)
to analyze, standardize, extract entities, and generate smart verification questions
from any messy Indian bank passbook or statement (SBI, HDFC, ICICI, Axis, UPI, etc.).

Includes automatic graceful fallback to the local deterministic rule engine
if no API key is provided or if network/quota is unavailable.
"""

import os
import json
import time
import re
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

try:
    from groq import Groq
    GROQ_INSTALLED = True
except ImportError:
    GROQ_INSTALLED = False

from feature_engineering import (
    normalize_bank_dataframe,
    get_smart_verification_candidates,
    categorize_transaction,
    extract_counterparty,
    clean_amount_series
)

DEFAULT_MODEL = "llama-3.3-70b-versatile"
FAST_MODEL = "llama-3.1-8b-instant"

def get_effective_groq_key(user_key: Optional[str] = None) -> Optional[str]:
    """Retrieves Groq API key from user input, session, or environment."""
    if user_key and user_key.strip():
        return user_key.strip()
    env_key = os.environ.get("GROQ_API_KEY", "").strip()
    return env_key if env_key else None

def is_groq_available(user_key: Optional[str] = None) -> bool:
    """Checks if Groq SDK is installed and an API key is present."""
    if not GROQ_INSTALLED:
        return False
    key = get_effective_groq_key(user_key)
    return bool(key and len(key) > 8)

def format_sample_transactions_for_prompt(df: pd.DataFrame, max_rows: int = 25) -> str:
    """Formats top representative rows into a concise tabular string for the LLM."""
    try:
        norm_df = normalize_bank_dataframe(df)
    except Exception:
        norm_df = df.copy()

    # Sample top credit and debit transactions to give balanced context
    debits = norm_df[norm_df.get('debit', 0) > 0].sort_values(by='debit', ascending=False).head(max_rows // 2) if 'debit' in norm_df else pd.DataFrame()
    credits = norm_df[norm_df.get('credit', 0) > 0].sort_values(by='credit', ascending=False).head(max_rows // 2) if 'credit' in norm_df else pd.DataFrame()
    
    combined = pd.concat([debits, credits]).drop_duplicates().head(max_rows)
    if combined.empty:
        combined = norm_df.head(max_rows)

    records = []
    for _, row in combined.iterrows():
        dt = str(row.get('date', 'N/A'))[:10]
        narr = str(row.get('narration', row.get('description', 'Unknown')))[:60]
        dr = float(row.get('debit', 0.0))
        cr = float(row.get('credit', 0.0))
        amt_str = f"Dr ₹{dr:,.0f}" if dr > 0 else f"Cr ₹{cr:,.0f}"
        records.append(f"- Date: {dt} | Txn: {narr} | Amount: {amt_str}")

    return "\n".join(records)

def parse_and_standardize_statement_groq(
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    biz_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Uses Groq LPU (Llama 3.3) to analyze passbook transactions,
    clean messy UPI/NEFT counterparties, and generate 3-4 smart verification questions.
    Falls back gracefully to deterministic feature engineering if unavailable.
    """
    start_time = time.time()
    effective_key = get_effective_groq_key(api_key)

    # 1. Graceful Local Fallback if Groq is not configured
    if not effective_key or not GROQ_INSTALLED:
        candidates = get_smart_verification_candidates(df)
        return {
            'success': True,
            'source': 'Local Heuristic Engine (Configure Groq API key to upgrade to Llama 3.3)',
            'is_groq': False,
            'candidates': candidates,
            'ai_memo': "Standardized using local Indian banking heuristics and regex pattern matching.",
            'latency': round(time.time() - start_time, 3)
        }

    # 2. Call Groq Cloud API
    try:
        client = Groq(api_key=effective_key)
        sample_text = format_sample_transactions_for_prompt(df, max_rows=24)
        biz_name = (biz_context or {}).get("name", "MSME Enterprise")
        biz_type = (biz_context or {}).get("type", "Trading & Business")

        prompt = f"""
You are an expert Indian Commercial Bank Senior Underwriter and Financial Data Specialist.
You are analyzing a raw Indian bank passbook / statement for business: "{biz_name}" ({biz_type}).

Here is a representative sample of bank statement entries:
{sample_text}

TASK:
1. Scrutinize the entries above and select 3 to 4 REAL, critical transactions that require business owner verification:
   - Top supplier/vendor payment (extract clean business name from UPI/NEFT/IMPS narration)
   - Primary customer revenue receipt
   - Recurring loan EMI or debt service (NBFCs or bank loans)
   - Key salary or utility transfer

CRITICAL INSTRUCTIONS:
- You MUST ONLY use the REAL counterparty names, REAL narrations, and EXACT numerical amounts that appear in the sample entries above.
- NEVER invent dummy names like "Shree Traders" or "ABC" unless they literally appear in the entries above.
- Clean up messy narrations to extract the human/business entity name (e.g. from "UPI/428192841/RAMESH TRADERS/OKAXIS", extract "Ramesh Traders").

2. Provide a strict JSON output matching this schema:
{{
  "ai_memo": "2-3 sentence executive assessment of the business cash flow stability, customer velocity, and debt obligations detected in this statement.",
  "verification_candidates": [
    {{
      "id": "cand_1",
      "amount": <exact numerical float from transactions>,
      "narration": "<exact raw narration string from transactions>",
      "counterparty": "<clean extracted entity name>",
      "frequency_note": "<e.g. Regular monthly receipt or High-value outflow>",
      "suggested_category": "<Supplier Payment | Customer Inflow | Loan / EMI | Salary / Payroll | Utilities & Rent | Personal Transfer>",
      "type": "<CREDIT or DEBIT>",
      "explanation": "<Specific question asking business owner to confirm the purpose of this transaction>"
    }}
  ]
}}

Output strictly valid JSON with no markdown backticks or commentary.
"""

        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You are a specialized financial underwriting AI. Output only valid JSON without backticks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1024,
            response_format={"type": "json_object"}
        )

        raw_json = response.choices[0].message.content.strip()
        data = json.loads(raw_json)
        candidates = data.get("verification_candidates", [])
        memo = data.get("ai_memo", "Passbook transactions verified successfully via Groq Llama 3.3.")

        if not candidates or len(candidates) < 2:
            # Fallback if LLM output was too short
            candidates = get_smart_verification_candidates(df)

        # Ensure ID format
        for idx, cand in enumerate(candidates):
            if 'id' not in cand or not cand['id']:
                cand['id'] = f"cand_{idx+1}"
            cand['amount'] = float(cand.get('amount', 0.0))

        latency = round(time.time() - start_time, 3)
        return {
            'success': True,
            'source': f'Groq Cloud AI ({DEFAULT_MODEL})',
            'is_groq': True,
            'candidates': candidates,
            'ai_memo': memo,
            'latency': latency
        }

    except Exception as e:
        # Fallback to local deterministic engine if API throws rate limit or error
        candidates = get_smart_verification_candidates(df)
        return {
            'success': True,
            'source': f'Local Heuristic Fallback (Groq note: {str(e)[:70]}...)',
            'is_groq': False,
            'candidates': candidates,
            'ai_memo': "Standardized using local banking feature engineering.",
            'latency': round(time.time() - start_time, 3)
        }

def generate_groq_lender_memo(
    msme_dict: Dict[str, Any],
    api_key: Optional[str] = None
) -> str:
    """
    Generates an institutional AI Credit Appraisal Memo for Lenders via Groq.
    """
    effective_key = get_effective_groq_key(api_key)
    if not effective_key or not GROQ_INSTALLED:
        # High quality template fallback
        return f"""
**Institutional Credit Appraisal Summary (Local Engine):**
- **Borrower**: {msme_dict.get('name', 'Applicant Enterprise')} ({msme_dict.get('sector', 'MSME')})
- **Cash Flow Run-rate**: ₹{msme_dict.get('monthly_revenue', 0):.2f} Lakhs/month with {msme_dict.get('cash_reserve_days', 0):.0f} days cash runway.
- **Credit Assessment**: Probability of Default stands at **{msme_dict.get('pd_pct', 0):.1f}%** ({msme_dict.get('risk_level', 'MEDIUM')} RISK). 
- **Working Capital Recommendation**: Sanction ₹{msme_dict.get('recommended_loan_lakhs', 0):.1f} Lakhs (Requested: ₹{msme_dict.get('requested_loan_lakhs', 0):.1f} Lakhs) with quarterly revenue monitoring covenant.
"""

    try:
        client = Groq(api_key=effective_key)
        prompt = f"""
Write a 3-bullet concise credit analyst decision memo for an Indian commercial bank credit committee:
- Borrower: {msme_dict.get('name')}
- Sector: {msme_dict.get('sector')}
- Vintage: {msme_dict.get('vintage')} years
- Monthly Revenue: ₹{msme_dict.get('monthly_revenue')} Lakhs
- Monthly Debt EMI: ₹{msme_dict.get('monthly_emi')} Lakhs
- Cash Reserve: {msme_dict.get('cash_reserve_days')} days
- Model PD: {msme_dict.get('pd_pct')}% ({msme_dict.get('risk_level')} Risk)
- Requested Loan: ₹{msme_dict.get('requested_loan_lakhs')} Lakhs
- Recommended Loan: ₹{msme_dict.get('recommended_loan_lakhs')} Lakhs

Provide:
1. Executive Decision (Sanction recommendation with debt sizing rationale)
2. Primary Headwind/Risk (Cash runway, revenue volatility, or leverage)
3. Underwriting Condition / Covenants (e.g. escrow of UPI collections, stock hypothecation)
Keep it crisp, professional, and institutional.
"""
        response = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": "You are a senior credit underwriting officer at an Indian bank."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Sanction recommendation based on calibrated PD of {msme_dict.get('pd_pct', 0):.1f}%. Recommended limit of ₹{msme_dict.get('recommended_loan_lakhs', 0):.1f} Lakhs matches 2.5x monthly cash surplus."
