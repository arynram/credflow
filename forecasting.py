"""
On-Demand Cash Flow Forecasting Module.
Addresses Fix #5:
  - Forecasts on demand for the selected MSME only (not precomputed for 1000s of rows).
  - Uses fast, robust Exponential Smoothing / Holt-Winters + 3-month Moving Average comparison.
  - Keeps strict "Insufficient History" guard when historical data is < 12 points.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

def forecast_cashflow(
    monthly_inflows: pd.Series, 
    forecast_horizon: int = 6, 
    min_history_required: int = 12
) -> Dict[str, Any]:
    """
    On-demand cash-flow forecaster with guards against thin time-series.
    """
    history_len = len(monthly_inflows)
    
    if history_len < min_history_required:
        return {
            "status": "GUARD_TRIGGERED",
            "message": f"Insufficient history: {history_len} monthly periods available. Minimum {min_history_required} months required for robust statistical forecasting.",
            "forecast_df": pd.DataFrame(),
            "ma_comparison": {}
        }

    series = monthly_inflows.values.astype(float)
    
    # 1. Baseline: 3-month Simple Moving Average
    ma_3m = float(np.mean(series[-3:]))
    ma_trend = float((series[-1] - series[-3]) / 2.0)

    # 2. Double Exponential Smoothing (Holt's Linear Trend)
    # Fast, analytical, no heavy external C++ binaries, highly effective on 12-36 points
    alpha = 0.35  # level smoothing
    beta = 0.15   # trend smoothing

    level = series[0]
    trend = series[1] - series[0] if history_len > 1 else 0.0

    for val in series[1:]:
        last_level = level
        level = alpha * val + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend

    # Projected future points
    forecast_points = []
    lower_bounds = []
    upper_bounds = []
    
    # Residual volatility estimate
    residuals = series[1:] - (series[:-1] + trend)
    sigma = float(np.std(residuals)) if len(residuals) > 2 else float(np.std(series) * 0.5)

    for h in range(1, forecast_horizon + 1):
        point_est = max(0.0, level + h * trend)
        uncertainty = 1.96 * sigma * np.sqrt(h) # 95% confidence band expanding with horizon
        forecast_points.append(round(point_est, 2))
        lower_bounds.append(round(max(0.0, point_est - uncertainty), 2))
        upper_bounds.append(round(point_est + uncertainty, 2))

    # Prepare timeline labels
    if isinstance(monthly_inflows.index, pd.PeriodIndex):
        last_period = monthly_inflows.index[-1]
        future_periods = [str(last_period + i) for i in range(1, forecast_horizon + 1)]
    else:
        future_periods = [f"M+{i}" for i in range(1, forecast_horizon + 1)]

    forecast_df = pd.DataFrame({
        "Period": future_periods,
        "Holt_Forecast": forecast_points,
        "Lower_95": lower_bounds,
        "Upper_95": upper_bounds,
        "Moving_Avg_3M_Baseline": [round(ma_3m + (h * ma_trend * 0.5), 2) for h in range(1, forecast_horizon + 1)]
    })

    return {
        "status": "SUCCESS",
        "message": f"Successfully generated {forecast_horizon}-month projection using Holt's Linear Trend with 3M-MA benchmark.",
        "forecast_df": forecast_df,
        "metrics": {
            "current_3m_run_rate": round(ma_3m, 2),
            "estimated_monthly_drift": round(trend, 2),
            "historical_months_used": history_len,
            "projected_6m_avg": round(float(np.mean(forecast_points)), 2)
        }
    }
