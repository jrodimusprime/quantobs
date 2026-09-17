"""Observations on a return stream. Descriptive only — nothing here judges.

Every function takes a periodic return series (and sometimes a benchmark or a
turnover series) and returns a number. No thresholds, no targets, no verdicts.

Conventions, fixed once so results stay comparable across experiments:

* `r` is a series of PERIODIC SIMPLE returns, already net of whatever costs the
  caller wants included. This library never applies a fee — costs are the
  caller's modelling decision and burying a default here would hide it.
* `periods` is the number of periods in a year (365 for daily crypto, 252 for
  daily equities, 12 for monthly). It is explicit everywhere, never guessed.
* Ratios are annualised. Distribution shape statistics are not.
* Anything undefined for the input returns `nan`, never a silent zero. A zero
  reads as a measurement; a nan reads as "this was not measurable", which is
  what it is.
"""
from __future__ import annotations

import numpy as np

DAILY_CRYPTO = 365
DAILY_EQUITY = 252
MONTHLY = 12


def _clean(r):
    r = np.asarray(r, float)
    return r[np.isfinite(r)]


def _equity(r):
    return np.cumprod(1.0 + r)


# --------------------------------------------------------------- return/risk
def total_return(r):
    r = _clean(r)
    return float(_equity(r)[-1] - 1.0) if len(r) else np.nan


def cagr(r, periods=DAILY_CRYPTO):
    r = _clean(r)
    if len(r) < 2:
        return np.nan
    end = _equity(r)[-1]
    if end <= 0:
        return -1.0
    return float(end ** (periods / len(r)) - 1.0)


def volatility(r, periods=DAILY_CRYPTO):
    r = _clean(r)
    return float(r.std(ddof=1) * np.sqrt(periods)) if len(r) > 1 else np.nan


def sharpe(r, periods=DAILY_CRYPTO, rf=0.0):
    """Annualised mean over annualised standard deviation."""
    r = _clean(r) - rf / periods
    if len(r) < 2 or r.std(ddof=1) <= 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods))


def sortino(r, periods=DAILY_CRYPTO, target=0.0):
    """Sharpe's denominator counts upside volatility as risk. This one doesn't.

    Two strategies with identical Sharpe can differ entirely in which tail
    produced the variance, and only one of them is a problem.
    """
    r = _clean(r) - target / periods
    down = r[r < 0]
    if len(r) < 2 or len(down) < 2:
        return np.nan
    dd = np.sqrt((down ** 2).mean())
    return float(r.mean() / dd * np.sqrt(periods)) if dd > 0 else np.nan


def sharpe_se(r=None, periods=DAILY_CRYPTO, s=None, n_periods=None):
    """Standard error of an annualised Sharpe, skew- and kurtosis-aware.

    Lo (2002), extended for non-normality. A Sharpe reported without this is a
    point estimate presented as a fact: over ~7 years of daily data the SE sits
    near 0.6, so differences smaller than that are not differences.
    """
    if r is not None:
        x = _clean(r)
        if len(x) < 8:
            return np.nan
        s = sharpe(x, periods) / np.sqrt(periods) * np.sqrt(periods)
        sp = sharpe(x, periods)
        g1, g2 = skew(x), excess_kurtosis(x)
        n = len(x)
        sp_per = sp / np.sqrt(periods)
        var = (1 + 0.5 * sp_per ** 2 - g1 * sp_per
               + (g2 / 4.0) * sp_per ** 2) / n
        return float(np.sqrt(max(var, 0.0)) * np.sqrt(periods))
    if s is None or not n_periods:
        return np.nan
    return float(np.sqrt((1.0 + 0.5 * (s / np.sqrt(periods)) ** 2) / n_periods)
                 * np.sqrt(periods))


# ------------------------------------------------------------- drawdown/pain
def max_drawdown(r):
    r = _clean(r)
    if not len(r):
        return np.nan
    eq = _equity(r)
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


def calmar(r, periods=DAILY_CRYPTO):
    dd = max_drawdown(r)
    return float(cagr(r, periods) / abs(dd)) if dd and dd < 0 else np.nan


def ulcer_index(r):
    """Root-mean-square drawdown: depth AND duration, not just the worst point.

    A -30% drawdown recovered in a week and one that lasted two years have the
    same max_drawdown. They are not the same experience and this separates them.
    """
    r = _clean(r)
    if not len(r):
        return np.nan
    eq = _equity(r)
    dd = eq / np.maximum.accumulate(eq) - 1.0
    return float(np.sqrt((dd ** 2).mean()))


def time_under_water(r, periods=DAILY_CRYPTO):
    """(longest, median) run below a prior peak, in YEARS.

    The number a person actually lives through. A max drawdown quoted without
    it describes a moment rather than a period.
    """
    r = _clean(r)
    if not len(r):
        return (np.nan, np.nan)
    eq = _equity(r)
    under = eq < np.maximum.accumulate(eq)
    runs, n = [], 0
    for u in under:
        if u:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    if not runs:
        return (0.0, 0.0)
    return (float(max(runs) / periods), float(np.median(runs) / periods))


# -------------------------------------------------------- distribution shape
def skew(r):
    r = _clean(r)
    if len(r) < 3 or r.std() == 0:
        return np.nan
    z = (r - r.mean()) / r.std()
    return float((z ** 3).mean())


def excess_kurtosis(r):
    r = _clean(r)
    if len(r) < 4 or r.std() == 0:
        return np.nan
    z = (r - r.mean()) / r.std()
    return float((z ** 4).mean() - 3.0)


def tail_ratio(r, q=0.05):
    """Right tail over left tail. 1.0 is symmetric; below 1.0 the losses win."""
    r = _clean(r)
    if len(r) < 20:
        return np.nan
    lo = abs(np.quantile(r, q))
    return float(np.quantile(r, 1 - q) / lo) if lo > 0 else np.nan


def concentration(r, top=(5, 20)):
    """Share of total return delivered by the best N periods.

    The cheapest test of whether something is a strategy or a lottery ticket.
    A pure-noise signal that scored Sharpe +1.85 in one experiment turned out
    to have LOWER concentration than the best of 84,585 engineered ones, which
    is not a comparison anyone had thought to make until this was measured.
    """
    r = _clean(r)
    tot = r.sum()
    if not len(r) or tot == 0:
        return {f"top{n}": np.nan for n in top}
    o = np.sort(r)[::-1]
    return {f"top{n}": float(o[:n].sum() / tot) for n in top}


# ------------------------------------------------------------- trade quality
def hit_rate(r):
    r = _clean(r)
    return float((r > 0).mean()) if len(r) else np.nan


def payoff_ratio(r):
    """Mean win over mean loss. Meaningless alone, essential beside hit_rate.

    A 76% hit rate with a 0.3 payoff loses money. Reporting either without the
    other is how win rates get mistaken for edges.
    """
    r = _clean(r)
    w, l = r[r > 0], r[r < 0]
    if not len(w) or not len(l):
        return np.nan
    return float(w.mean() / abs(l.mean()))


def profit_factor(r):
    r = _clean(r)
    gains, losses = r[r > 0].sum(), abs(r[r < 0].sum())
    return float(gains / losses) if losses > 0 else np.nan


def expectancy(r):
    """Mean return per period. hit_rate * payoff collapses to this."""
    r = _clean(r)
    return float(r.mean()) if len(r) else np.nan


def edge_per_turnover(r, turnover, periods=DAILY_CRYPTO):
    """Basis points earned per unit of notional traded.

    The only observation here denominated in the SAME UNITS AS COST, which is
    what makes it the one that can be compared to a fee directly. A strategy
    turning over 119x/yr for a 64% return earns ~57bps per unit traded; against
    a 2bps taker fee that is 28x cover. Against a 60bps fee it is nothing.
    """
    r, t = _clean(r), _clean(turnover)
    if not len(r) or not len(t) or t.sum() <= 0:
        return np.nan
    yrs = len(r) / periods
    gross = np.exp(np.log1p(r).sum() / yrs) - 1.0
    return float(gross / (t.sum() / yrs) * 10000.0)


# ------------------------------------------------------- benchmark relation
def alpha_beta(r, benchmark, periods=DAILY_CRYPTO):
    """(annualised alpha, beta). Correlation says direction; beta says size."""
    r, b = np.asarray(r, float), np.asarray(benchmark, float)
    m = np.isfinite(r) & np.isfinite(b)
    r, b = r[m], b[m]
    if len(r) < 8 or b.std() == 0:
        return (np.nan, np.nan)
    beta = float(np.cov(r, b, ddof=1)[0, 1] / b.var(ddof=1))
    return (float((r.mean() - beta * b.mean()) * periods), beta)


def correlation(r, benchmark):
    r, b = np.asarray(r, float), np.asarray(benchmark, float)
    m = np.isfinite(r) & np.isfinite(b)
    if m.sum() < 8 or r[m].std() == 0 or b[m].std() == 0:
        return np.nan
    return float(np.corrcoef(r[m], b[m])[0, 1])


# ------------------------------------------------------------ sample quality
def effective_n(r, max_lag=40):
    """Independent observations, discounting autocorrelation.

    A thousand daily returns from a book held for fifty days at a time are not
    a thousand independent observations. Every uncertainty computed from a raw
    count is overconfident by the ratio this reports.
    """
    r = _clean(r)
    n = len(r)
    if n < 20 or r.std() == 0:
        return float(n)
    x = r - r.mean()
    s = 0.0
    for k in range(1, min(max_lag, n // 4) + 1):
        a = float(np.corrcoef(x[:-k], x[k:])[0, 1])
        if not np.isfinite(a) or a <= 0:
            break
        s += a
    return float(n / (1.0 + 2.0 * s))


def rolling_stability(r, window=182, periods=DAILY_CRYPTO):
    """(mean, sd, share positive) of rolling-window Sharpes.

    One number for a seven-year backtest hides whether the edge was steady or
    arrived in a single quarter.
    """
    r = _clean(r)
    if len(r) < window * 2:
        return {"mean": np.nan, "sd": np.nan, "pos": np.nan}
    vals = [sharpe(r[i:i + window], periods)
            for i in range(0, len(r) - window, max(window // 2, 1))]
    vals = np.array([v for v in vals if np.isfinite(v)])
    if not len(vals):
        return {"mean": np.nan, "sd": np.nan, "pos": np.nan}
    return {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1))
            if len(vals) > 1 else np.nan, "pos": float((vals > 0).mean())}


# ------------------------------------------- multiple-testing aware measures
def probabilistic_sharpe(r, periods=DAILY_CRYPTO, benchmark_sr=0.0):
    """P(true Sharpe > benchmark_sr), adjusted for skew and kurtosis.

    Bailey & Lopez de Prado (2012). Answers "how confident can I be this is
    above zero at all", which a bare Sharpe cannot.
    """
    x = _clean(r)
    if len(x) < 20:
        return np.nan
    sr = sharpe(x, periods) / np.sqrt(periods)
    g1, g2 = skew(x), excess_kurtosis(x)
    n = len(x)
    denom = np.sqrt(max(1 - g1 * sr + (g2 / 4.0) * sr ** 2, 1e-12))
    z = (sr - benchmark_sr / np.sqrt(periods)) * np.sqrt(n - 1) / denom
    return float(0.5 * (1.0 + _erf(z / np.sqrt(2.0))))


def deflated_sharpe(r, n_trials, periods=DAILY_CRYPTO, trial_sr_var=None):
    """Probabilistic Sharpe against the Sharpe the BEST OF n_trials reaches by
    chance. Bailey & Lopez de Prado (2014).

    The single most important observation here for any result that came out of
    a search. A Sharpe of +1.76 selected from 4.8M attempts and a Sharpe of
    +1.76 from one pre-registered hypothesis are not the same observation, and
    only this distinguishes them.

    `trial_sr_var` is the variance of the PER-PERIOD Sharpe ACROSS the trials
    that were searched. Pass it when the search's own spread is known -- that is
    the quantity the formula actually calls for. It defaults to the sampling
    variance of this series' own Sharpe, which is the right order of magnitude
    when trials are independent draws on the same data and is the usual
    published simplification.
    """
    x = _clean(r)
    if len(x) < 20 or n_trials is None or n_trials < 2:
        return np.nan
    sr = sharpe(x, periods) / np.sqrt(periods)          # per period
    if trial_sr_var is None:
        trial_sr_var = (1.0 + 0.5 * sr ** 2) / len(x)
    if not np.isfinite(trial_sr_var) or trial_sr_var <= 0:
        return np.nan
    e = 0.5772156649015329                               # Euler-Mascheroni
    z1 = _ppf(1.0 - 1.0 / n_trials)
    z2 = _ppf(1.0 - 1.0 / (n_trials * np.e))
    sr0 = np.sqrt(trial_sr_var) * ((1 - e) * z1 + e * z2)
    return probabilistic_sharpe(x, periods,
                                benchmark_sr=sr0 * np.sqrt(periods))


def expected_max_sharpe(n_trials, n_periods, periods=DAILY_CRYPTO,
                        trial_sr_var=None, sr=0.0):
    """The ANNUALISED Sharpe the best of n_trials reaches by chance alone.

    The same quantity deflated_sharpe compares against, exposed directly
    because it is the number a person actually wants when asking "how good does
    this have to be before it means anything at this sample size".
    """
    if n_trials is None or n_trials < 2 or n_periods < 2:
        return np.nan
    if trial_sr_var is None:
        trial_sr_var = (1.0 + 0.5 * (sr / np.sqrt(periods)) ** 2) / n_periods
    e = 0.5772156649015329
    z1, z2 = _ppf(1.0 - 1.0 / n_trials), _ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(trial_sr_var) * ((1 - e) * z1 + e * z2)
                 * np.sqrt(periods))


def _erf(x):
    # Abramowitz & Stegun 7.1.26; keeps the package numpy-only.
    s = np.sign(x)
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def _ppf(p):
    # Acklam's inverse normal CDF, adequate to ~1e-9.
    if not 0 < p < 1:
        return np.nan
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > ph:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    rr = q * q
    return (((((a[0] * rr + a[1]) * rr + a[2]) * rr + a[3]) * rr + a[4]) * rr + a[5]) * q / \
           (((((b[0] * rr + b[1]) * rr + b[2]) * rr + b[3]) * rr + b[4]) * rr + 1)
