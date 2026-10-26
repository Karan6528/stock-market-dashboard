"""Unit tests for the maths and data layer. Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from core import data, indicators as ind, ml, risk


@pytest.fixture(scope="module")
def prices():
    return data.demo_prices("TEST", "2018-01-01", "2024-12-31")


# ----------------------------------------------------------------- indicators
def test_rsi_bounds_and_extremes(prices):
    r = ind.rsi(prices["Close"]).dropna()
    assert r.between(0, 100).all()
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert ind.rsi(up).iloc[-1] == pytest.approx(100)
    down = pd.Series(np.arange(60, 1, -1, dtype=float))
    assert ind.rsi(down).iloc[-1] == pytest.approx(0)


def test_macd_histogram_is_difference(prices):
    m = ind.macd(prices["Close"])
    assert np.allclose(m["Histogram"], m["MACD"] - m["Signal"])


def test_bollinger_ordering(prices):
    b = ind.bollinger(prices["Close"]).dropna()
    assert (b["Upper"] >= b["Middle"]).all() and (b["Middle"] >= b["Lower"]).all()


def test_stochastic_bounds(prices):
    s = ind.stochastic(prices).dropna()
    assert s["%K"].between(0, 100).all()


# ----------------------------------------------------------------------- risk
def test_drawdown_known_series():
    s = pd.Series([100, 120, 90, 130, 65.0])
    assert risk.max_drawdown(s) == pytest.approx(-0.5)


def test_var_and_cvar():
    rets = pd.Series(np.linspace(-0.10, 0.10, 201))
    var = risk.var_historical(rets, 0.95)
    assert var == pytest.approx(-0.09, abs=1e-3)
    assert risk.cvar_historical(rets, 0.95) <= var


def test_beta_of_scaled_series_aligns_dates():
    idx = pd.bdate_range("2023-01-01", periods=300)
    m = pd.Series(np.random.default_rng(0).normal(0, 0.01, 300), index=idx)
    a = 1.5 * m
    assert risk.beta(a.iloc[20:], m) == pytest.approx(1.5)  # different lengths on purpose


def test_optimiser_weights_valid(prices):
    closes = pd.DataFrame({s: data.demo_prices(s, "2020-01-01", "2024-12-31")["Close"] for s in "ABCD"})
    rets = risk.returns(closes)
    for obj in ("max_sharpe", "min_vol"):
        w = risk.optimise(rets, obj)
        assert w.sum() == pytest.approx(1) and (w >= -1e-9).all()
    w_min = risk.optimise(rets, "min_vol")
    equal = np.full(4, 0.25)
    assert risk.annual_vol(risk.portfolio_returns(rets, w_min)) <= risk.annual_vol(risk.portfolio_returns(rets, equal)) + 1e-9


def test_monte_carlo_shape():
    paths = risk.monte_carlo_paths(100, pd.Series(np.full(100, 0.001)), days=10, n_paths=50)
    assert paths.shape == (11, 50) and (paths[0] == 100).all()


# ------------------------------------------------------------------------- ml
def test_features_have_no_lookahead(prices):
    """Changing future prices must not change today's features."""
    cut = 1000
    base = ml.build_features(prices).iloc[:cut]
    shocked = prices.copy()
    shocked.iloc[cut:, :4] *= 3
    assert np.allclose(base.fillna(0), ml.build_features(shocked).iloc[:cut].fillna(0))


def test_target_is_next_day_return(prices):
    X, y, close_t, _ = ml.make_dataset(prices)
    d = y.index[100]
    pos = prices.index.get_loc(d)
    assert y.loc[d] == pytest.approx(prices["Close"].iloc[pos + 1] / prices["Close"].iloc[pos] - 1)


def test_evaluate_includes_naive_baseline(prices):
    res = ml.evaluate(prices, ["Ridge Regression"], 0.2)
    assert "Naive (no change)" in res["metrics"].index
    assert res["n_test"] > 100


# ----------------------------------------------------------------------- data
def test_demo_data_is_deterministic_and_valid():
    a = data.demo_prices("XYZ", "2022-01-01", "2022-12-31")
    b = data.demo_prices("XYZ", "2022-06-01", "2022-12-31")
    assert a.loc[b.index[0], "Close"] == pytest.approx(b["Close"].iloc[0])  # same history whatever the window
    assert (a["High"] >= a[["Open", "Close"]].max(axis=1)).all()
    assert (a["Low"] <= a[["Open", "Close"]].min(axis=1)).all()


def test_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PRICE_DIR", tmp_path)
    df = data.demo_prices("RT", "2023-01-01", "2023-03-31")
    data.save_prices("RT", df)
    data.save_prices("RT", df.iloc[-10:])  # merging must not duplicate rows
    back = data.load_cached_prices("RT")
    assert len(back) == len(df)
    assert np.allclose(back["Close"], df["Close"])


def test_parse_symbols():
    assert data.parse_symbols(" aapl, MSFT ,,aapl; ^gspc") == ["AAPL", "MSFT", "^GSPC"]


# -------------------------------------------------------------------- planner
from core import planner, sentiment  # noqa: E402


def test_position_size_respects_risk_and_cash():
    p = planner.position_size(10_000, 100.0, 2.0, 0.02, 2.0, 2.0)
    assert p.stop == pytest.approx(96.0) and p.target == pytest.approx(108.0)
    assert p.shares == 50 and p.max_loss == pytest.approx(200.0)  # 2% of 10k
    tight = planner.position_size(1_000, 100.0, 0.1, 0.5, 2.0, 2.0)
    assert tight.shares == 10 and tight.limited_by == "available cash"


def test_historical_outcomes_known_path():
    idx = pd.bdate_range("2024-01-01", periods=8)
    close = pd.Series([100, 101, 103, 106, 111, 100, 100, 100.0], index=idx)
    df = pd.DataFrame({"Close": close, "High": close, "Low": close})
    out = planner.historical_outcomes(df, 0.10, 0.05, 4)
    first = out.iloc[0]
    assert first["Outcome"] == "Target" and first["Days"] == 4


def test_price_reaction_after_close_uses_next_day():
    idx = pd.bdate_range("2024-01-01", periods=5)
    px = pd.Series([100, 102, 99, 101, 103.0], index=idx)
    day, r1, _ = sentiment.price_reaction(pd.Timestamp("2024-01-02 18:00"), px)
    assert day == idx[2] and r1 == pytest.approx(99 / 102 - 1)
    day, r1, _ = sentiment.price_reaction(pd.Timestamp("2024-01-02 10:00"), px)
    assert day == idx[1] and r1 == pytest.approx(0.02)


def test_parse_published_formats():
    assert sentiment.parse_published("20251025T130000").day == 25
    assert sentiment.parse_published("2025-10-25T17:00:00Z").hour == 13  # converted to New York time
    assert pd.isna(sentiment.parse_published(""))
