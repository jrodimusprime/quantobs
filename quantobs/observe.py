"""`observe()` and `render()` — the fixed observation set and its fixed layout.

Two experiments that both call `observe()` produce comparable records without
anyone deciding which numbers to include. That decision is made once, here, and
is the reason this package is separate and changes rarely.

Nothing in this module judges. `render()` prints a floor or a target only if the
CALLER passes one in `context`, and even then it is printed beside the number,
never used to colour it pass or fail.
"""
from __future__ import annotations

import numpy as np

from . import core as c

# Fixed order. Appending is safe; reordering or renaming breaks comparability
# with every record already written, so it is a breaking change.
FIELDS = [
    ("sharpe",          "Sharpe"),
    ("sharpe_se",       "  +/- SE"),
    ("prob_sharpe",     "P(Sharpe>0)"),
    ("deflated_sharpe", "deflated (vs best-of-N)"),
    ("sortino",         "Sortino"),
    ("cagr",            "CAGR"),
    ("volatility",      "volatility"),
    ("max_drawdown",    "max drawdown"),
    ("calmar",          "Calmar"),
    ("ulcer",           "ulcer index"),
    ("tuw_max",         "longest underwater"),
    ("tuw_median",      "median underwater"),
    ("skew",            "skew"),
    ("excess_kurtosis", "excess kurtosis"),
    ("tail_ratio",      "tail ratio"),
    ("top5",            "top 5 periods = share of return"),
    ("top20",           "top 20 periods"),
    ("hit_rate",        "hit rate"),
    ("payoff_ratio",    "payoff ratio"),
    ("profit_factor",   "profit factor"),
    ("edge_bps",        "edge per unit traded"),
    ("beta",            "beta to benchmark"),
    ("alpha",           "alpha to benchmark"),
    ("correlation",     "correlation to benchmark"),
    ("n_periods",       "periods observed"),
    ("effective_n",     "effective (independent) periods"),
    ("roll_mean",       "rolling Sharpe, mean"),
    ("roll_sd",         "rolling Sharpe, sd"),
    ("roll_pos",        "rolling windows positive"),
]

_PCT = {"cagr", "volatility", "max_drawdown", "top5", "top20", "hit_rate",
        "roll_pos", "ulcer"}
_INT = {"n_periods", "effective_n"}
_YRS = {"tuw_max", "tuw_median"}


def observe(returns, benchmark=None, turnover=None, n_trials=None,
            periods=c.DAILY_CRYPTO, window=182):
    """Every observation this library makes, as a plain dict.

    `n_trials` is how many strategies were tried before this one was selected.
    Pass it whenever the result came out of a search — without it the deflated
    Sharpe cannot be computed, and a searched result reported without a deflated
    Sharpe is the error this whole package exists to make visible.
    """
    r = np.asarray(returns, float)
    conc = c.concentration(r)
    tuw_max, tuw_med = c.time_under_water(r, periods)
    roll = c.rolling_stability(r, window, periods)
    a, b = (c.alpha_beta(r, benchmark, periods) if benchmark is not None
            else (np.nan, np.nan))
    return {
        "sharpe": c.sharpe(r, periods),
        "sharpe_se": c.sharpe_se(r, periods),
        "prob_sharpe": c.probabilistic_sharpe(r, periods),
        "deflated_sharpe": (c.deflated_sharpe(r, n_trials, periods)
                            if n_trials else np.nan),
        "n_trials": n_trials,
        "sortino": c.sortino(r, periods),
        "cagr": c.cagr(r, periods),
        "volatility": c.volatility(r, periods),
        "max_drawdown": c.max_drawdown(r),
        "calmar": c.calmar(r, periods),
        "ulcer": c.ulcer_index(r),
        "tuw_max": tuw_max, "tuw_median": tuw_med,
        "skew": c.skew(r), "excess_kurtosis": c.excess_kurtosis(r),
        "tail_ratio": c.tail_ratio(r),
        "top5": conc.get("top5"), "top20": conc.get("top20"),
        "hit_rate": c.hit_rate(r), "payoff_ratio": c.payoff_ratio(r),
        "profit_factor": c.profit_factor(r),
        "edge_bps": (c.edge_per_turnover(r, turnover, periods)
                     if turnover is not None else np.nan),
        "alpha": a, "beta": b,
        "correlation": (c.correlation(r, benchmark)
                        if benchmark is not None else np.nan),
        "n_periods": float(np.isfinite(r).sum()),
        "effective_n": c.effective_n(r),
        "roll_mean": roll["mean"], "roll_sd": roll["sd"], "roll_pos": roll["pos"],
    }


def _fmt(k, v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "not measurable"
    if k in _PCT:
        return f"{v * 100:+.1f}%" if k not in ("hit_rate", "roll_pos", "top5",
                                               "top20", "ulcer") \
            else f"{v * 100:.1f}%"
    if k in _INT:
        return f"{v:,.0f}"
    if k in _YRS:
        return f"{v:.2f}y"
    if k == "edge_bps":
        return f"{v:.1f} bps"
    if k in ("prob_sharpe", "deflated_sharpe"):
        return f"{v:.3f}"
    return f"{v:+.2f}"


def render(obs, label="", context=None, indent="  "):
    """Fixed layout. `context` is printed beside a value, never used to judge it.

        render(obs, context={"edge_bps": "vs 2bps taker",
                             "deflated_sharpe": "selected from 4.8M"})
    """
    context = context or {}
    out = [f"{label}"] if label else []
    width = max(len(t) for _, t in FIELDS) + 2
    for key, title in FIELDS:
        if key not in obs:
            continue
        note = context.get(key, "")
        out.append(f"{title:<{width}}{_fmt(key, obs[key]):>16}"
                   + (f"   {note}" if note else ""))
    return "\n".join(indent + ln for ln in out)


def row(obs, label, keys=("sharpe", "deflated_sharpe", "sortino", "calmar",
                          "max_drawdown", "edge_bps", "top5", "correlation")):
    """One line, for tables. Same keys every time unless explicitly overridden."""
    return f"  {label:<30}" + "".join(f"{_fmt(k, obs.get(k)):>16}" for k in keys)


def row_header(keys=("sharpe", "deflated_sharpe", "sortino", "calmar",
                     "max_drawdown", "edge_bps", "top5", "correlation")):
    short = {"sharpe": "Sharpe", "deflated_sharpe": "deflated", "sortino": "Sortino",
             "calmar": "Calmar", "max_drawdown": "maxDD", "edge_bps": "edge/turn",
             "top5": "top5 share", "correlation": "corr"}
    return f"  {'strategy':<30}" + "".join(f"{short.get(k, k):>16}" for k in keys)
