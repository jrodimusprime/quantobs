"""Properties that must hold. Not targets -- these are definitional."""
import numpy as np
import quantobs as qo

rng = np.random.default_rng(0)


def test_sharpe_scales_with_periods():
    r = rng.normal(0.001, 0.01, 2000)
    assert abs(qo.sharpe(r, 365) / qo.sharpe(r, 252) - np.sqrt(365 / 252)) < 1e-9


def test_sortino_exceeds_sharpe_when_downside_is_small():
    r = np.concatenate([rng.normal(0.004, 0.004, 900), rng.normal(-0.001, 0.001, 100)])
    assert qo.sortino(r) > qo.sharpe(r)


def test_max_drawdown_is_non_positive():
    assert qo.max_drawdown(rng.normal(0, 0.02, 500)) <= 0


def test_concentration_of_one_big_day():
    r = np.zeros(200); r[7] = 0.5
    assert abs(qo.concentration(r)["top5"] - 1.0) < 1e-9


def test_effective_n_falls_with_autocorrelation():
    w = rng.normal(0, 1, 4000)
    ac = np.convolve(w, np.ones(20) / 20, "same")     # strongly autocorrelated
    assert qo.effective_n(ac) < qo.effective_n(w) / 3


def test_deflated_sharpe_falls_as_trials_rise():
    # a MARGINAL series: a strong one saturates both ends at 1.0 and the test
    # would pass vacuously while telling you nothing
    r = rng.normal(0.0004, 0.01, 2500)
    assert qo.deflated_sharpe(r, 10) > qo.deflated_sharpe(r, 1_000_000)


def test_deflated_below_probabilistic():
    r = rng.normal(0.0015, 0.01, 2500)
    assert qo.deflated_sharpe(r, 1000) <= qo.probabilistic_sharpe(r)


def test_edge_per_turnover_units():
    r = np.full(365, 0.001)                 # ~44% a year
    t = np.full(365, 0.1)                   # 36.5x turnover a year
    assert 100 < qo.edge_per_turnover(r, t) < 200


def test_undefined_returns_nan_not_zero():
    assert np.isnan(qo.sharpe([0.01]))
    assert np.isnan(qo.payoff_ratio(np.array([0.01, 0.02])))


def test_time_under_water_zero_for_monotonic():
    assert qo.time_under_water(np.full(500, 0.001))[0] == 0.0


def test_observe_has_every_declared_field():
    obs = qo.observe(rng.normal(0.001, 0.01, 1200),
                     benchmark=rng.normal(0, 0.02, 1200),
                     turnover=np.full(1200, 0.05), n_trials=5000)
    for key, _ in qo.FIELDS:
        assert key in obs, key


def test_no_function_returns_a_verdict():
    """The design constraint: this package measures, it never judges.

    Checked behaviourally rather than by grepping for words -- an earlier
    version of this test failed on the word "floor" appearing in a docstring
    that existed to say floors are NOT here.
    """
    r = rng.normal(0.001, 0.01, 800)
    b = rng.normal(0.0, 0.02, 800)
    t = np.full(800, 0.05)
    for name in qo.__all__:
        fn = getattr(qo, name)
        if not callable(fn) or name in ("observe", "render", "row", "row_header"):
            continue
        for args in ((r,), (r, b), (r, t), (r, 1000)):
            try:
                out = fn(*args)
            except Exception:
                continue
            flat = out.values() if isinstance(out, dict) else (
                out if isinstance(out, tuple) else (out,))
            for v in flat:
                assert not isinstance(v, (bool, str)), f"{name} returned {v!r}"
            break


def test_expected_max_sharpe_grows_with_trials():
    a = qo.expected_max_sharpe(10, 2500)
    b = qo.expected_max_sharpe(1_000_000, 2500)
    assert b > a > 0
