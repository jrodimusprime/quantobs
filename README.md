# quantobs

Canonical definitions for **observations on trading performance**.

This library answers *"what is true about this return stream?"* — never
*"is that good?"*. It contains no thresholds, no targets, no verdicts and no
notion of passing. Every function returns a number; the caller decides what the
number means.

That separation is the point. Judgement changes per experiment. Measurement
should not, and when it drifts the results of different experiments stop being
comparable.

## What is deliberately absent

- **Targets and floors.** No "Sharpe above 1.5", no "keep 70% after dropping a
  coin". Those belong to an experiment's pre-registered gate, not here.
- **Verdict vocabulary.** Nothing returns "pass", "fail" or "winner".
- **Benchmark opinions.** `alpha_beta` will decompose against whatever series
  you hand it. It has no view on which benchmark is the right one.
- **Fee assumptions.** Costs are an argument, never a default buried in a
  function.

## Why several of these exist

Most of this set was added because its absence caused a specific error:

| Metric | The mistake it prevents |
|---|---|
| `sharpe_se` | Comparing a Sharpe gap to the wrong uncertainty. A Sharpe over 6.7y carries an SE near 0.6; treating a 0.001 gap as meaningful follows directly from not having this. |
| `deflated_sharpe` | Reporting the best of N searched strategies as if it were one hypothesis. |
| `skew`, `excess_kurtosis` | Assuming normality. A null with skew +1.08 and kurtosis +1.59 makes a Gaussian max formula understate its own tail. |
| `concentration` | Calling something a strategy when five days carry 30% of its return. |
| `time_under_water` | Reporting a −24% max drawdown without saying it lasted fourteen months. |
| `edge_per_turnover` | Reporting a pre-cost number as the headline. This is the only metric denominated in the same units as cost. |
| `effective_n` | Counting correlated observations as independent. |

## Install

```bash
pip install -e ~/code/quantobs
```

## Use

```python
import quantobs as qo

obs = qo.observe(returns, benchmark=btc, turnover=turn, n_trials=4_800_000)
print(qo.render(obs, label="xs_clv* lb21 br3 rb1"))
```

`observe()` returns a plain dict. `render()` formats it in a fixed field order
so two experiments print comparably. Neither says whether it is any good.
