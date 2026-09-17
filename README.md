# quantobs

Canonical definitions for **observations on trading performance**.

This library answers *"what is true about this return stream?"* — never
*"is that good?"*. It contains no thresholds, no targets, no verdicts and no
notion of passing. Every function returns a number; the caller decides what the
number means.

That separation is the point. Judgement changes per experiment. Measurement
should not, and when it drifts the results of different experiments stop being
comparable.

```bash
pip install -e ~/code/quantobs
```

```python
import quantobs as qo

obs = qo.observe(returns, benchmark=btc, turnover=turn, n_trials=8_589_986)
print(qo.render(obs, label="xs_clv* lb21 br3 rb1"))
```

---

## Conventions

Fixed once, so records stay comparable across experiments.

| | |
|---|---|
| **Returns** | `r` is a series of **periodic simple returns**, already net of whatever costs the caller chose to include. This library never applies a fee — cost is a modelling decision and burying a default here would hide it. |
| **Annualisation** | `periods` is periods-per-year and is **explicit everywhere, never guessed**. `DAILY_CRYPTO = 365`, `DAILY_EQUITY = 252`, `MONTHLY = 12`. |
| **What is annualised** | Ratios are. Distribution-shape statistics are not. |
| **Undefined** | Returns `nan`, never a silent `0`. A zero reads as a measurement; a nan reads as *"this was not measurable"*, which is what it is. |
| **Verdicts** | No function returns a `bool` or a `str`. Enforced by a test that walks every public function, not by convention. |

---

## The observations

`observe()` returns all 29 fields below as a plain dict. Every one is also
available as a standalone function.

### Return and risk

| Field | Function | What it measures | How to read it |
|---|---|---|---|
| `sharpe` | `sharpe(r, periods, rf=0)` | Annualised mean over annualised standard deviation. | The headline number, and the most over-trusted one in the field. Never read it without `sharpe_se`. |
| `sharpe_se` | `sharpe_se(r, periods)` | Standard error of that Sharpe, skew- and kurtosis-aware (Lo 2002). | Differences smaller than this are not differences. Over ~7 years of daily data it sits near **0.6**. |
| `sortino` | `sortino(r, periods, target=0)` | As Sharpe, but the denominator counts only downside deviation. | Two strategies with identical Sharpe can differ entirely in which tail produced the variance. Only one of them is a problem. |
| `cagr` | `cagr(r, periods)` | Compound annual growth rate. | Geometric, so it already carries volatility drag. |
| `volatility` | `volatility(r, periods)` | Annualised standard deviation. | |
| — | `total_return(r)` | Cumulative return over the whole series. | Not in `observe()`; for equity-curve endpoints. |
| — | `expectancy(r)` | Mean return per period. | `hit_rate × payoff_ratio` collapses to this. |

### Drawdown and pain

| Field | Function | What it measures | How to read it |
|---|---|---|---|
| `max_drawdown` | `max_drawdown(r)` | Worst peak-to-trough decline. | A single moment. Says nothing about how long it lasted. |
| `calmar` | `calmar(r, periods)` | CAGR over absolute max drawdown. | Return per unit of worst-case pain. |
| `ulcer` | `ulcer_index(r)` | Root-mean-square drawdown across the whole series. | Depth **and** duration. A −30% drawdown recovered in a week and one lasting two years have the same `max_drawdown`; they are not the same experience. |
| `tuw_max` | `time_under_water(r, periods)[0]` | Longest unbroken run below a prior peak, in **years**. | The number a person actually lives through. |
| `tuw_median` | `time_under_water(r, periods)[1]` | Median such run, in years. | A large gap between median and max means the pain sits in one episode. |

### Distribution shape

| Field | Function | What it measures | How to read it |
|---|---|---|---|
| `skew` | `skew(r)` | Third standardised moment. | `0` is symmetric. Positive means the right tail is longer. |
| `excess_kurtosis` | `excess_kurtosis(r)` | Fourth standardised moment minus 3. | `0` is Gaussian. Large positive values mean any formula assuming normality is mis-specified for this series. |
| `tail_ratio` | `tail_ratio(r, q=0.05)` | 95th percentile over absolute 5th percentile. | `1.0` is symmetric. Below `1.0`, the losing tail is the larger one. |
| `top5` | `concentration(r)["top5"]` | Share of total return delivered by the best 5 periods. | The cheapest test of whether something is a strategy or a lottery ticket. |
| `top20` | `concentration(r)["top20"]` | Same for the best 20. | |

### Trade quality

| Field | Function | What it measures | How to read it |
|---|---|---|---|
| `hit_rate` | `hit_rate(r)` | Fraction of periods that were positive. | **Meaningless alone.** Always read beside `payoff_ratio`. |
| `payoff_ratio` | `payoff_ratio(r)` | Mean win over mean loss. | A 76% hit rate with a 0.3 payoff loses money. |
| `profit_factor` | `profit_factor(r)` | Gross gains over gross losses. | `1.0` is break-even by construction. |
| `edge_bps` | `edge_per_turnover(r, turnover, periods)` | Basis points earned per unit of notional traded. | **The only observation here denominated in the same units as cost**, which is what makes it directly comparable to a fee. A book turning over 119×/yr for a 64% return earns ~57bps per unit traded. |

### Relation to a benchmark

| Field | Function | What it measures | How to read it |
|---|---|---|---|
| `correlation` | `correlation(r, benchmark)` | Pearson correlation. | Direction of co-movement. |
| `beta` | `alpha_beta(r, b, periods)[1]` | Regression slope against the benchmark. | **Size** of exposure, which correlation does not give you. |
| `alpha` | `alpha_beta(r, b, periods)[0]` | Annualised intercept. | Return not explained by benchmark exposure. |

The library has no opinion on which benchmark is the right one. It decomposes
against whatever series you hand it.

### Sample quality

| Field | Function | What it measures | How to read it |
|---|---|---|---|
| `n_periods` | — | Count of finite observations. | |
| `effective_n` | `effective_n(r, max_lag=40)` | Independent observations after discounting positive autocorrelation. | A thousand daily returns from a book held fifty days at a time are not a thousand independent observations. Any uncertainty computed from the raw count is overconfident by this ratio. |
| `roll_mean` | `rolling_stability(r, window, periods)["mean"]` | Mean of rolling-window Sharpes. | |
| `roll_sd` | `…["sd"]` | Standard deviation of those. | An `sd` near the `mean` says the edge arrives in bursts rather than steadily. |
| `roll_pos` | `…["pos"]` | Share of rolling windows that were positive. | One number for a seven-year backtest hides whether the edge was steady or arrived in a single quarter. |

### Multiple-testing aware

A result that came out of a search is a different observation from the same
number arrived at once. These are what make that difference visible.

| Field | Function | What it measures | How to read it |
|---|---|---|---|
| `prob_sharpe` | `probabilistic_sharpe(r, periods, benchmark_sr=0)` | P(true Sharpe > benchmark), adjusted for skew and kurtosis. Bailey & López de Prado (2012). | Answers *"how confident can I be this is above zero at all"*, which a bare Sharpe cannot. |
| `deflated_sharpe` | `deflated_sharpe(r, n_trials, periods, trial_sr_var=None)` | `prob_sharpe` measured against the Sharpe the **best of `n_trials`** reaches by chance. Bailey & López de Prado (2014). | **The single most important field here for any searched result.** A Sharpe of +1.76 selected from 4.8M attempts and the same number from one pre-registered hypothesis are not the same observation, and only this distinguishes them. |
| — | `expected_max_sharpe(n_trials, n_periods, periods, …)` | The annualised Sharpe chance alone reaches at a given search size. | The quantity `deflated_sharpe` compares against, exposed directly because it is what you actually want when asking *how good does this have to be before it means anything at this sample size*. |

`trial_sr_var` is the variance of the per-period Sharpe **across the trials that
were searched**. Pass it when the search's own spread is known — that is the
quantity the formula calls for. It defaults to the sampling variance of the
series' own Sharpe, the usual published simplification.

> **Pass `n_trials` whenever the result came out of a search.** Without it the
> deflated Sharpe cannot be computed, and a searched result reported without one
> is the error this package exists to make visible.

---

## Formatting

| Function | Purpose |
|---|---|
| `observe(returns, benchmark, turnover, n_trials, periods, window)` | All 29 fields as a dict. |
| `render(obs, label, context)` | Fixed field order, one per line. `context` annotates a value; it never colours it pass or fail. |
| `row(obs, label, keys)` / `row_header(keys)` | One line per strategy, for tables. Same eight keys every time unless overridden. |
| `FIELDS` | The field order itself. Appending is safe; **reordering or renaming is a breaking change**, because it breaks comparability with every record already written. |

A rendered record looks like this:

```
xs_clv* lb21 br3 rb1
Sharpe                                      +1.94
  +/- SE                                    +0.39   differences smaller than this are not differences
P(Sharpe>0)                                 1.000
deflated (vs best-of-N)                     0.393   selected from 8.59M trials
Sortino                                     +2.12
CAGR                                       +60.6%
volatility                                 +26.2%
max drawdown                               -27.6%
Calmar                                      +2.20
ulcer index                                  7.1%
longest underwater                          0.67y
median underwater                           0.01y
skew                                        +0.39
excess kurtosis                            +19.99
tail ratio                                  +1.28
top 5 periods = share of return             16.0%
top 20 periods                              42.9%
hit rate                                    51.6%
payoff ratio                                +1.29
profit factor                               +1.40
edge per unit traded                     67.2 bps   vs 2bps MEXC taker
beta to benchmark                           +0.01
alpha to benchmark                          +0.50
correlation to benchmark                    +0.03
periods observed                            2,434
effective (independent) periods             1,858
rolling Sharpe, mean                        +1.58
rolling Sharpe, sd                          +1.64
rolling windows positive                    88.0%
```

Read together, those last two lines say the edge is not steady, and
`deflated_sharpe` says chance alone reaches +2.04 at that search size. Neither
statement is in the Sharpe.

---

## Why several of these exist

Each was added because its absence caused a specific, traceable error.

| Metric | The mistake it prevents |
|---|---|
| `sharpe_se` | Comparing a Sharpe gap to the wrong uncertainty. A gap of **0.0013** was treated as a pass when the SE was 0.6. |
| `deflated_sharpe` | Reporting the best of 4.8M searched strategies as if it were one hypothesis. |
| `expected_max_sharpe` | Not knowing chance alone reaches **+2.04** at 8.6M trials, which made "+1.94" look like a finding. |
| `skew`, `excess_kurtosis` | Assuming normality. A null with skew +1.08 makes a Gaussian max formula understate its own tail — and the patch for that let a single outlier draw set the threshold for an entire research programme. |
| `concentration` | Calling something a strategy when five days carry 30% of its return. A pure-noise signal once beat the best of 84,585 engineered ones on this measure. |
| `time_under_water`, `ulcer_index` | Reporting a −24% max drawdown without saying how long it lasted. |
| `edge_per_turnover` | Reporting a pre-cost number as the headline. |
| `effective_n` | Counting correlated observations as independent. |
| `hit_rate` + `payoff_ratio` | Mistaking a win rate for an edge. |
| `rolling_stability` | Presenting a seven-year average as though the edge were steady. |

---

## Tests

13 tests covering definitional properties — that Sharpe scales as
`sqrt(periods)`, that `effective_n` falls under autocorrelation, that
`deflated_sharpe` falls as trials rise, that undefined inputs give `nan` rather
than `0`.

One is a design test rather than a numerical one: it walks every public function
and fails if any returns a `bool` or a `str`. An earlier version grepped the
source for the word *"floor"* and failed on a docstring that existed to say
floors are **not** here — testing prose is not testing design.

```bash
python -m pytest tests -q
```
