import pandas as pd
import random

def validate_bank_integrity(df: pd.DataFrame) -> dict:
    """
    Validates structural and financial integrity of the bank statement.
    """
    total_tx = len(df)
    if total_tx < 50:
        return {"valid": False, "reason": f"Insufficient data: Found only {total_tx} transactions (Minimum 50 required)."}
    
    if 'date' in df:
        days_span = (df['date'].max() - df['date'].min()).days
        if days_span < 30:
            return {"valid": False, "reason": f"Insufficient history: Statement covers only {days_span} days (Minimum 3 months required for reliable scoring)."}

    return {"valid": True, "reason": "Bank Data Integrity Check Passed"}

def calculate_verification_score(bank_valid: bool, has_gst: bool, has_itr: bool, has_udyam: bool, gst_discrepancy: float) -> dict:
    """
    Calculates Multi-Source Financial Verification Score.
    """
    sources_provided = 1 # Bank is mandatory
    max_sources = 4
    
    if has_gst: sources_provided += 1
    if has_itr: sources_provided += 1
    if has_udyam: sources_provided += 1

    base_score = 0
    if bank_valid: base_score += 25
    
    # GST Logic
    if has_gst:
        if gst_discrepancy <= 0.10: base_score += 25
        elif gst_discrepancy <= 0.25: base_score += 15
        else: base_score += 5
    else:
        # Scale score up if source not provided so we don't punish them, but confidence drops.
        pass
        
    score = int((base_score / (sources_provided * 25)) * 100) if sources_provided > 0 else 0
    
    # Add identity checks
    if has_udyam: score = min(100, score + 15)
    
    confidence = "Low"
    if sources_provided == 2: confidence = "Moderate"
    elif sources_provided >= 3: confidence = "High"

    return {
        "score": score,
        "coverage": f"{sources_provided} of {max_sources} sources",
        "confidence": confidence,
        "sources_provided": sources_provided
    }
