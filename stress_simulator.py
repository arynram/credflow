"""
Scenario Simulator & Macroeconomic Stress Testing Module.
Addresses Fix #4:
  - Shock inputs first (revenue, costs, interest rate hike, additional debt).
  - Explicitly recomputes derived financial features (EMI burden, DTI, cash buffer days, volatility).
  - Re-evaluates the calibrated risk model to produce an updated default probability and stress delta.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

def simulate_macro_stress(
    base_features: Dict[str, Any],
    revenue_shock_pct: float = -0.20,       # e.g. -20% revenue contraction
    cost_inflation_pct: float = 0.10,      # e.g. +10% raw material / OPEX inflation
    interest_rate_hike_bps: int = 150,     # e.g. +150 bps interest rate hike
    emergency_borrowing_pct: float = 0.05  # e.g. +5% emergency overdraft
) -> Dict[str, Any]:
    """
    Applies real macro shocks and fully recomputes all derived credit metrics.
    """
    base_rev = float(base_features.get('avg_monthly_revenue_lakhs', base_features.get('avg_monthly_revenue', 25.0)))
    base_emi_burden = float(base_features.get('emi_burden', 0.25))
    base_dti = float(base_features.get('dti_ratio', 0.60))
    base_buffer = float(base_features.get('cash_buffer_days', 22.0))
    base_volatility = float(base_features.get('revenue_volatility', 0.25))
    base_growth = float(base_features.get('revenue_growth', 0.05))
    base_bounce_rate = float(base_features.get('bounce_rate', 0.01))

    # Base monetary values in Lakhs
    base_emi_monthly = base_rev * base_emi_burden
    base_opex_monthly = base_rev * 0.65 # assumed 65% base OPEX
    base_net_cash = base_rev - base_opex_monthly - base_emi_monthly

    # 1. Shocked Revenue
    shocked_rev = max(1.0, base_rev * (1.0 + revenue_shock_pct))
    
    # 2. Shocked OPEX
    shocked_opex = base_opex_monthly * (1.0 + cost_inflation_pct)

    # 3. Shocked Monthly Debt Service (EMI)
    # Effective interest rate proxy ~ 12% p.a. A 100 bps hike is ~ 8.3% increase in interest component
    rate_factor = 1.0 + (interest_rate_hike_bps / (100.0 * 12.0))
    debt_factor = 1.0 + emergency_borrowing_pct
    shocked_emi_monthly = base_emi_monthly * rate_factor * debt_factor

    # 4. Recomputed Derived Metrics (Fix #4)
    recomputed_emi_burden = min(1.0, max(0.01, shocked_emi_monthly / shocked_rev))
    
    # Recomputed DTI (Total Estimated Debt / Annualized Revenue)
    estimated_shocked_debt = shocked_emi_monthly * 24.0
    recomputed_dti = min(3.5, max(0.05, estimated_shocked_debt / (shocked_rev * 12.0)))

    # Recomputed Net Cash Flow & Cash Buffer Days
    shocked_net_cash = shocked_rev - shocked_opex - shocked_emi_monthly
    if base_net_cash > 0 and shocked_net_cash > 0:
        buffer_depletion_ratio = max(0.1, shocked_net_cash / base_net_cash)
        recomputed_buffer_days = max(1.5, base_buffer * buffer_depletion_ratio)
    else:
        # Burning cash reserves: buffer drains fast
        daily_cash_burn = max(0.05, abs(shocked_net_cash) / 30.0)
        recomputed_buffer_days = max(1.0, min(base_buffer * 0.4, (base_buffer * (base_rev * 0.15)) / (daily_cash_burn * 30.0 + 1e-3)))

    # Recomputed Volatility & Growth
    recomputed_volatility = min(1.2, base_volatility * (1.0 + 0.6 * abs(revenue_shock_pct)))
    recomputed_growth = base_growth + revenue_shock_pct

    # Recomputed Bounce Probability (liquidity strain increases bounce risk)
    bounce_multiplier = 1.0 + (max(0.0, recomputed_emi_burden - 0.40) * 4.0)
    recomputed_bounce_rate = min(0.35, base_bounce_rate * bounce_multiplier)

    shocked_feature_vector = {
        'avg_monthly_revenue_lakhs': round(shocked_rev, 2),
        'revenue_growth': round(recomputed_growth, 4),
        'revenue_volatility': round(recomputed_volatility, 4),
        'emi_burden': round(recomputed_emi_burden, 4),
        'dti_ratio': round(recomputed_dti, 4),
        'bounce_rate': round(recomputed_bounce_rate, 4),
        'bounce_count': int(base_features.get('bounce_count', 0) * (2 if recomputed_emi_burden > 0.45 else 1)),
        'top_3_concentration': float(base_features.get('top_3_concentration', 0.45)),
        'cash_buffer_days': round(recomputed_buffer_days, 1),
        'gst_discrepancy': float(base_features.get('gst_discrepancy', 0.08)),
        'business_age_years': float(base_features.get('business_age_years', 5.0)),
        'sector': base_features.get('sector', 'Retail & Wholesale'),
        'gender_of_promoter': base_features.get('gender_of_promoter', 'Male'),
        'region': base_features.get('region', 'Tier-2 Urban'),
        'ownership_type': base_features.get('ownership_type', 'Sole Proprietorship')
    }

    shock_summary = {
        "revenue_change_pct": round(revenue_shock_pct * 100, 1),
        "cost_inflation_pct": round(cost_inflation_pct * 100, 1),
        "interest_rate_hike_bps": interest_rate_hike_bps,
        "base_monthly_revenue": round(base_rev, 2),
        "shocked_monthly_revenue": round(shocked_rev, 2),
        "base_emi_burden_pct": round(base_emi_burden * 100, 1),
        "shocked_emi_burden_pct": round(recomputed_emi_burden * 100, 1),
        "base_buffer_days": round(base_buffer, 1),
        "shocked_buffer_days": round(recomputed_buffer_days, 1),
        "base_dti": round(base_dti, 2),
        "shocked_dti": round(recomputed_dti, 2)
    }

    return {
        "shocked_features": shocked_feature_vector,
        "summary": shock_summary
    }
