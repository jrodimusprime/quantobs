"""quantobs — canonical observations on trading performance.

Descriptive only. No thresholds, no targets, no verdicts. See README.
"""
from .core import (DAILY_CRYPTO, DAILY_EQUITY, MONTHLY, alpha_beta, cagr,
                   calmar, concentration, correlation, deflated_sharpe,
                   edge_per_turnover, effective_n, excess_kurtosis, expectancy,
                   expected_max_sharpe,
                   hit_rate, max_drawdown, payoff_ratio, probabilistic_sharpe,
                   profit_factor, rolling_stability, sharpe, sharpe_se, skew,
                   sortino, tail_ratio, time_under_water, total_return,
                   ulcer_index, volatility)
from .observe import FIELDS, observe, render, row, row_header

__version__ = "1.0.0"
__all__ = [n for n in dir() if not n.startswith("_")]
