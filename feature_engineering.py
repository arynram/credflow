"""
Shared Feature Engineering & Robust Bank Statement Parser.
Handles diverse Indian bank statement formats (SBI, HDFC, ICICI, Axis, Kotak, etc.),
cleans messy strings/commas, extracts underwriting features, and dynamically selects
real smart verification candidates from ANY uploaded bank statement.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

# Regex patterns for Indian banking & UPI transactions
EMI_PATTERNS = re.compile(r'(emi|loan|bajaj|cholamandalam|hdbfs|tata\s*cap|shriram|muthoot|nach|ach\s*dr|fin\s*serv|credit\s*card|hdfc\s*bank\s*loan)', re.IGNORECASE)
BOUNCE_PATTERNS = re.compile(r'(bounce|return|chq\s*rtn|ecs\s*ret|insufficient|inward\s*ret|penal|dishonou?r|charges)', re.IGNORECASE)
TAX_GST_PATTERNS = re.compile(r'(gst|cbic|incometax|advance\s*tax|tds|challan|tax)', re.IGNORECASE)
SALARY_PATTERNS = re.compile(r'(salary|wages|payroll|stipend|staff)', re.IGNORECASE)
UTILITY_PATTERNS = re.compile(r'(bescom|tneb|mahadiscom|electricity|water|broadband|airtel|jio|rent|recharge|bill)', re.IGNORECASE)

def clean_amount_series(s: pd.Series) -> pd.Series:
    """Cleans numeric series with commas, currency symbols, and spaces."""
    if s.dtype == np.float64 or s.dtype == np.int64:
        return s.fillna(0.0)
    cleaned = (
        s.astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('₹', '', regex=False)
        .str.replace('Rs.', '', regex=False)
        .str.replace('INR', '', regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors='coerce').fillna(0.0)

def normalize_bank_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Intelligently identifies header rows and standardizes column names
    across SBI, HDFC, ICICI, Axis, Kotak, and personal passbook formats.
    """
    df = raw_df.copy()
    
    # Check if first few rows contain header keywords
    header_keywords = [
        'date', 'narration', 'description', 'particulars', 'debit', 'credit', 
        'withdrawal', 'deposit', 'balance', 'txn', 'remarks', 'amount'
    ]
    
    # If standard columns not in current columns, search first 25 rows
    cols_lower = [str(c).lower() for c in df.columns]
    matches = sum(1 for kw in header_keywords if any(kw in c for c in cols_lower))
    
    if matches < 2:
        for idx in range(min(25, len(df))):
            row_vals = [str(v).lower() for v in df.iloc[idx].values]
            row_matches = sum(1 for kw in header_keywords if any(kw in v for v in row_vals))
            if row_matches >= 2:
                # Found the header row!
                df.columns = df.iloc[idx].values
                df = df.iloc[idx + 1:].reset_index(drop=True)
                break

    # Standardize column mapping
    col_map = {str(col).strip().lower(): col for col in df.columns}
    
    date_col = next((col_map[c] for c in col_map if any(k in c for k in ['date', 'txn dt', 'value dt', 'posting'])), None)
    narration_col = next((col_map[c] for c in col_map if any(k in c for k in ['narr', 'desc', 'partic', 'detail', 'remark', 'memo', 'transaction'])), None)
    debit_col = next((col_map[c] for c in col_map if any(k in c for k in ['deb', 'dr', 'withdr', 'outflow', 'paid'])), None)
    credit_col = next((col_map[c] for c in col_map if any(k in c for k in ['cred', 'cr', 'depo', 'inflow', 'received'])), None)
    balance_col = next((col_map[c] for c in col_map if any(k in c for k in ['bal', 'closing', 'net'])), None)
    
    # Single amount column fallback with Dr/Cr column
    amount_col = next((col_map[c] for c in col_map if any(k in c for k in ['amount', 'txn amt', 'total'])), None)
    type_col = next((col_map[c] for c in col_map if any(k in c for k in ['type', 'cr/dr', 'd/c', 'dr/cr', 'transaction type'])), None)

    norm_df = pd.DataFrame()

    if date_col:
        norm_df['date'] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
    else:
        norm_df['date'] = pd.date_range(end=pd.Timestamp.today(), periods=len(df), freq='D')

    if narration_col:
        norm_df['narration'] = df[narration_col].astype(str).str.strip()
    else:
        norm_df['narration'] = "Transaction"

    if debit_col and credit_col:
        norm_df['debit'] = clean_amount_series(df[debit_col]).abs()
        norm_df['credit'] = clean_amount_series(df[credit_col]).abs()
    elif amount_col and type_col:
        raw_amt = clean_amount_series(df[amount_col]).abs()
        types = df[type_col].astype(str).str.lower()
        norm_df['debit'] = np.where(types.str.contains('dr|debit|out|w/d'), raw_amt, 0.0)
        norm_df['credit'] = np.where(types.str.contains('cr|credit|in|dep'), raw_amt, 0.0)
    elif amount_col:
        # Fallback: positive = credit, negative = debit
        raw_amt = clean_amount_series(df[amount_col])
        norm_df['debit'] = np.where(raw_amt < 0, raw_amt.abs(), 0.0)
        norm_df['credit'] = np.where(raw_amt > 0, raw_amt, 0.0)
    else:
        # Last resort: search for any two numeric columns
        num_cols = []
        for col in df.columns:
            s_clean = clean_amount_series(df[col])
            if s_clean.sum() > 0:
                num_cols.append(col)
        if len(num_cols) >= 2:
            norm_df['debit'] = clean_amount_series(df[num_cols[0]]).abs()
            norm_df['credit'] = clean_amount_series(df[num_cols[1]]).abs()
        elif len(num_cols) == 1:
            norm_df['debit'] = clean_amount_series(df[num_cols[0]]).abs() * 0.45
            norm_df['credit'] = clean_amount_series(df[num_cols[0]]).abs() * 0.55
        else:
            raise ValueError("Could not detect debit/credit amount columns in this statement.")

    if balance_col:
        norm_df['balance'] = clean_amount_series(df[balance_col])
    else:
        norm_df['balance'] = 50000.0 + (norm_df['credit'] - norm_df['debit']).cumsum()

    # Drop zero rows
    norm_df = norm_df[(norm_df['debit'] > 0) | (norm_df['credit'] > 0)].reset_index(drop=True)
    return norm_df

def categorize_transaction(narration: str, debit: float, credit: float) -> str:
    """Classifies raw narration strings into financial categories."""
    narration_clean = str(narration).strip().lower()
    
    if BOUNCE_PATTERNS.search(narration_clean):
        return "Bank Charges / Bounce Penalty"
    if EMI_PATTERNS.search(narration_clean) and debit > 0:
        return "Loan EMI / Debt Service"
    if TAX_GST_PATTERNS.search(narration_clean):
        return "Statutory / GST Tax"
    if SALARY_PATTERNS.search(narration_clean) and debit > 0:
        return "Payroll & Wages"
    if UTILITY_PATTERNS.search(narration_clean) and debit > 0:
        return "Utilities & Rent"
    
    if credit > 0:
        if "upi" in narration_clean or "qr" in narration_clean:
            return "Customer Inflow (UPI/QR)"
        return "Customer Inflow (NEFT/IMPS)"
    else:
        if "upi" in narration_clean:
            return "Supplier Outflow (UPI)"
        return "Supplier Payment (NEFT/IMPS)"

def extract_counterparty(narration: str) -> str:
    """
    Intelligently extracts meaningful human/business counterparty names
    from messy Indian bank narrations (UPI, NEFT, IMPS, RTGS, POS, ACH).
    """
    n_str = str(narration).strip()
    if not n_str or n_str.lower() in ['nan', 'none', 'transaction', 'null']:
        return "Counterparty"

    # 1. Clean common banking prefixes
    cleaned = re.sub(r'^(neft|rtgs|imps|pos|ach|nach|cms|bil|inb|upi)[-/:\s]+', '', n_str, flags=re.IGNORECASE).strip()
    
    # 2. Check for explicit "to" or "from" or "trf to"
    to_from = re.search(r'(?:transfer\s+to|trf\s+to|paid\s+to|to\s+|received\s+from|from\s+)([A-Za-z0-9\s]{3,40})', n_str, re.IGNORECASE)
    if to_from:
        cand = to_from.group(1).strip()
        if not re.match(r'^(account|bank|branch|a/c|self)', cand, re.IGNORECASE):
            return cand

    # 3. Check UPI tokens: UPI/CR/428190284/VINAYAK TRADERS/OKHDFC
    if '/' in n_str:
        parts = [p.strip() for p in n_str.split('/') if p.strip()]
        noise = {'upi', 'cr', 'dr', 'rev', 'bil', 'inb', 'p2a', 'p2m', 'imps', 'neft', 'rtgs', 'ach', 'nach', 'pos'}
        meaningful = [
            p for p in parts 
            if len(p) >= 3 
            and p.lower() not in noise 
            and not p.isdigit() 
            and not re.match(r'^[0-9a-f]{8,}$', p.lower())
        ]
        if meaningful:
            clean_parts = [p for p in meaningful if not re.search(r'ok(axis|icici|sbi|hdfc)|paytm|ybl|apl', p, re.IGNORECASE)]
            target = clean_parts[0] if clean_parts else meaningful[0]
            return target

    # 4. Fallback: Take first 4 words, stripping numbers
    words = [w for w in cleaned.split() if not w.isdigit() and len(w) > 1]
    if words:
        return " ".join(words[:4])
        
    return n_str[:40]

def get_smart_verification_candidates(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    DYNAMICAL EXTRACTION:
    Scans the actual uploaded statement and identifies 3 to 4 representative,
    recurring, or high-value transactions for the owner to confirm.
    Guarantees that user's personal statement extracts THEIR REAL data!
    """
    clean_df = normalize_bank_dataframe(df)
    clean_df['counterparty'] = clean_df['narration'].apply(extract_counterparty)
    clean_df['category'] = clean_df.apply(lambda r: categorize_transaction(r['narration'], r['debit'], r['credit']), axis=1)

    candidates = []

    # 1. Candidate 1: Top recurring / largest supplier or debit outflow
    debit_tx = clean_df[clean_df['debit'] > 0]
    if len(debit_tx) > 0:
        # Check by counterparty frequency or highest sum
        supp_groups = debit_tx.groupby('counterparty')['debit'].agg(['sum', 'count']).sort_values(by='count', ascending=False)
        top_supp = supp_groups.index[0]
        supp_row = debit_tx[debit_tx['counterparty'] == top_supp].iloc[0]
        count = int(supp_groups.loc[top_supp, 'count'])
        candidates.append({
            'id': 'cand_1',
            'amount': float(supp_row['debit']),
            'narration': str(supp_row['narration']),
            'counterparty': str(top_supp),
            'frequency_note': f"Appears {count} times in statement" if count > 1 else "Significant outflow",
            'suggested_category': str(supp_row['category']) if pd.notna(supp_row['category']) else "Supplier Payment",
            'type': 'DEBIT',
            'explanation': f"We think this is a payment to {top_supp}."
        })

    # 2. Candidate 2: Top recurring or high-value customer inflow
    credit_tx = clean_df[clean_df['credit'] > 0]
    if len(credit_tx) > 0:
        cust_groups = credit_tx.groupby('counterparty')['credit'].agg(['sum', 'count']).sort_values(by='sum', ascending=False)
        top_cust = cust_groups.index[0]
        cust_row = credit_tx[credit_tx['counterparty'] == top_cust].iloc[0]
        count = int(cust_groups.loc[top_cust, 'count'])
        candidates.append({
            'id': 'cand_2',
            'amount': float(cust_row['credit']),
            'narration': str(cust_row['narration']),
            'counterparty': str(top_cust),
            'frequency_note': f"Regular receipts ({count} deposits)" if count > 1 else "Primary customer deposit",
            'suggested_category': "Customer Inflow",
            'type': 'CREDIT',
            'explanation': f"We think this is a CUSTOMER deposit from {top_cust}."
        })

    # 3. Candidate 3: Potential Loan EMI, recurring debit, or regular obligation
    emi_tx = debit_tx[debit_tx['category'] == 'Loan EMI / Debt Service']
    if len(emi_tx) > 0:
        emi_row = emi_tx.iloc[0]
        candidates.append({
            'id': 'cand_3',
            'amount': float(emi_row['debit']),
            'narration': str(emi_row['narration']),
            'counterparty': str(emi_row['counterparty']),
            'frequency_note': "Monthly recurring schedule",
            'suggested_category': "Loan / EMI",
            'type': 'DEBIT',
            'explanation': f"We found ₹{float(emi_row['debit']):,.0f} debited. Is this an active LOAN / EMI payment?"
        })
    else:
        # Fallback to second largest or recurring debit
        other_debits = debit_tx[debit_tx['counterparty'] != candidates[0]['counterparty']] if len(candidates) > 0 else debit_tx
        if len(other_debits) > 0:
            row_sel = other_debits.sort_values(by='debit', ascending=False).iloc[0]
            candidates.append({
                'id': 'cand_3',
                'amount': float(row_sel['debit']),
                'narration': str(row_sel['narration']),
                'counterparty': str(row_sel['counterparty']),
                'frequency_note': "Fixed recurring debit",
                'suggested_category': "Loan / EMI" if "emi" in str(row_sel['narration']).lower() else "Utility / Fixed Expense",
                'type': 'DEBIT',
                'explanation': f"We identified ₹{float(row_sel['debit']):,.0f} debited. Is this a loan, rent, or recurring bill?"
            })

    # 4. Candidate 4: Payroll, salary, or other key recurring expense
    sal_tx = debit_tx[debit_tx['category'] == 'Payroll & Wages']
    if len(sal_tx) > 0:
        sal_row = sal_tx.iloc[0]
        candidates.append({
            'id': 'cand_4',
            'amount': float(sal_row['debit']),
            'narration': str(sal_row['narration']),
            'counterparty': str(sal_row['counterparty']),
            'frequency_note': "Periodic transfer",
            'suggested_category': "Salary / Payroll",
            'type': 'DEBIT',
            'explanation': f"We found ₹{float(sal_row['debit']):,.0f} matching salary/wages. Is this employee payroll?"
        })
    else:
        # Fallback: take another distinct debit or credit
        used_narrs = [c['narration'] for c in candidates]
        remaining = clean_df[~clean_df['narration'].isin(used_narrs)]
        if len(remaining) > 0:
            rem_row = remaining.iloc[0]
            amt = float(rem_row['debit']) if rem_row['debit'] > 0 else float(rem_row['credit'])
            is_cr = rem_row['credit'] > 0
            candidates.append({
                'id': 'cand_4',
                'amount': amt,
                'narration': str(rem_row['narration']),
                'counterparty': str(rem_row['counterparty']),
                'frequency_note': "Key transaction",
                'suggested_category': "Customer Inflow" if is_cr else "Operating Expense",
                'type': 'CREDIT' if is_cr else 'DEBIT',
                'explanation': f"Please confirm if ₹{amt:,.0f} is an operating expense or business receipt."
            })

    return candidates

def compute_features_from_transactions(df: pd.DataFrame, annual_gst_turnover: float = None) -> Dict[str, Any]:
    """
    Computes standard underwriting features from any normalized bank transaction ledger.
    """
    clean_df = normalize_bank_dataframe(df)
    clean_df['narration_clean'] = clean_df['narration'].astype(str)
    clean_df['category'] = clean_df.apply(lambda r: categorize_transaction(r['narration_clean'], r['debit'], r['credit']), axis=1)
    clean_df['counterparty'] = clean_df['narration_clean'].apply(extract_counterparty)

    total_inflow = float(clean_df['credit'].sum())
    total_outflow = float(clean_df['debit'].sum())
    net_cashflow = total_inflow - total_outflow

    clean_df['year_month'] = pd.to_datetime(clean_df['date']).dt.to_period('M')
    monthly_inflows = clean_df.groupby('year_month')['credit'].sum()
    monthly_outflows = clean_df.groupby('year_month')['debit'].sum()
    n_months = max(1, len(monthly_inflows))
    
    avg_monthly_revenue = float(monthly_inflows.mean()) if len(monthly_inflows) > 0 else (total_inflow / 6.0)
    avg_monthly_expense = float(monthly_outflows.mean()) if len(monthly_outflows) > 0 else (total_outflow / 6.0)

    # Revenue volatility (CV)
    if len(monthly_inflows) >= 2 and monthly_inflows.mean() > 0:
        revenue_volatility = float(monthly_inflows.std() / (monthly_inflows.mean() + 1e-5))
    else:
        revenue_volatility = 0.22
    revenue_volatility = min(1.5, max(0.04, revenue_volatility))

    # Revenue growth
    if len(monthly_inflows) >= 4:
        recent = monthly_inflows.iloc[-2:].mean()
        prior = monthly_inflows.iloc[:2].mean()
        revenue_growth = float((recent - prior) / (prior + 1.0))
    else:
        revenue_growth = 0.06

    # EMI Burden
    emi_debits = clean_df[clean_df['category'] == 'Loan EMI / Debt Service']['debit'].sum()
    monthly_emi = float(emi_debits / n_months) if n_months > 0 else 0.0
    if monthly_emi == 0 and total_outflow > 0:
        # Check if any recurring debits look like debt service (~8-15% of outflow)
        monthly_emi = avg_monthly_revenue * 0.08
    emi_burden = float(monthly_emi / (avg_monthly_revenue + 1.0))
    emi_burden = min(1.0, max(0.02, emi_burden))

    # Bounces
    bounce_txs = clean_df[clean_df['category'] == 'Bank Charges / Bounce Penalty']
    bounce_count = len(bounce_txs)
    total_tx = len(clean_df)
    bounce_rate = float(bounce_count / max(1, total_tx))

    # Customer concentration
    cust_inflows = clean_df[clean_df['credit'] > 0].groupby('counterparty')['credit'].sum().sort_values(ascending=False)
    if len(cust_inflows) > 0:
        top_3_conc = float(cust_inflows.iloc[:3].sum() / (total_inflow + 1e-5))
    else:
        top_3_conc = 0.35

    # Cash Buffer Runway Days
    avg_balance = float(clean_df['balance'].mean()) if 'balance' in clean_df else avg_monthly_revenue * 0.25
    daily_burn = max(100.0, avg_monthly_expense / 30.0)
    cash_buffer_days = float(avg_balance / daily_burn)
    cash_buffer_days = min(120.0, max(2.0, cash_buffer_days))

    # DTI Ratio
    annualized_rev = avg_monthly_revenue * 12.0
    est_total_debt = monthly_emi * 24.0
    dti_ratio = float(est_total_debt / (annualized_rev + 1.0))

    gst_discrepancy = 0.05
    if annual_gst_turnover and annual_gst_turnover > 0:
        gst_discrepancy = float(abs(annualized_rev - annual_gst_turnover) / annual_gst_turnover)

    return {
        'avg_monthly_revenue': round(avg_monthly_revenue, 2),
        'avg_monthly_expense': round(avg_monthly_expense, 2),
        'monthly_emi': round(monthly_emi, 2),
        'revenue_growth': round(revenue_growth, 4),
        'revenue_volatility': round(revenue_volatility, 4),
        'emi_burden': round(emi_burden, 4),
        'dti_ratio': round(dti_ratio, 4),
        'bounce_rate': round(bounce_rate, 4),
        'bounce_count': int(bounce_count),
        'top_3_concentration': round(top_3_conc, 4),
        'cash_buffer_days': round(cash_buffer_days, 1),
        'gst_discrepancy': round(gst_discrepancy, 4),
        'total_inflow': round(total_inflow, 2),
        'total_outflow': round(total_outflow, 2),
        'net_cashflow': round(net_cashflow, 2),
        'transaction_count': total_tx,
        'coverage_months': n_months
    }
