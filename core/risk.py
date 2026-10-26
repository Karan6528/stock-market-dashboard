"""Return, risk and portfolio maths. Pure functions, easy to unit test."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import TRADING_DAYS


def returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return prices.pct_change().dropna(how="all")


def cagr(prices: pd.Series) -> float:
    prices = prices.dropna()
    if len(prices) < 2 or prices.iloc[0] <= 0:
        return float("nan")
    years = max((prices.index[-1] - prices.index[0]).days / 365.25, 1 / 365.25)
    return float((prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1)


def annual_return(rets: pd.Series) -> float:
    """Arithmetic mean daily return, annualised (what mean-variance theory uses)."""
    return float(rets.mean() * TRADING_DAYS)


def annual_vol(rets: pd.Series) -> float:
    return float(rets.std() * np.sqrt(TRADING_DAYS))


def sharpe(rets: pd.Series, rf_annual: float = 0.0) -> float:
    excess = rets - rf_annual / TRADING_DAYS
    sd = excess.std()
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def sortino(rets: pd.Series, rf_annual: float = 0.0) -> float:
    excess = rets - rf_annual / TRADING_DAYS
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean())
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else float("nan")


def drawdown(prices_or_equity: pd.Series) -> pd.Series:
    running_max = prices_or_equity.cummax()
    return prices_or_equity / running_max - 1


def max_drawdown(prices_or_equity: pd.Series) -> float:
    return float(drawdown(prices_or_equity).min())


def var_historical(rets: pd.Series, confidence: float = 0.95) -> float:
    """Daily historical VaR as a (negative) return, e.g. -0.025 = -2.5%."""
    return float(np.percentile(rets.dropna(), (1 - confidence) * 100))


def cvar_historical(rets: pd.Series, confidence: float = 0.95) -> float:
    var = var_historical(rets, confidence)
    tail = rets[rets <= var]
    return float(tail.mean()) if len(tail) else var


def beta(asset_rets: pd.Series, market_rets: pd.Series) -> float:
    """Beta on the dates both series share (the original code mixed lengths)."""
    joined = pd.concat([asset_rets, market_rets], axis=1, join="inner").dropna()
    if len(joined) < 10:
        return float("nan")
    cov = np.cov(joined.iloc[:, 0], joined.iloc[:, 1], ddof=1)
    return float(cov[0, 1] / cov[1, 1])


def portfolio_returns(rets: pd.DataFrame, weights) -> pd.Series:
    w = np.asarray(weights, dtype=float)
    return rets.dot(w)


def monte_carlo_paths(last_price: float, daily_rets: pd.Series, days: int = 252,
                      n_paths: int = 1000, seed: int | None = 42) -> np.ndarray:
    """Geometric Brownian motion using the historical log-return mean and stdev.
    Returns an array of shape (days + 1, n_paths), first row = last price."""
    log_rets = np.log1p(daily_rets.dropna())
    mu, sigma = log_rets.mean(), log_rets.std()
    rng = np.random.default_rng(seed)
    shocks = rng.normal(mu, sigma, size=(days, n_paths))
    paths = last_price * np.exp(np.cumsum(shocks, axis=0))
    return np.vstack([np.full(n_paths, last_price), paths])


def percentile_bands(paths: np.ndarray, pcts=(5, 25, 50, 75, 95)) -> pd.DataFrame:
    return pd.DataFrame({f"p{p}": np.percentile(paths, p, axis=1) for p in pcts})


# --------------------------------------------------------------------------- #
# portfolio optimisation
# --------------------------------------------------------------------------- #
def _stats(w, mean, cov, rf):
    ret = float(np.dot(w, mean) * TRADING_DAYS)
    vol = float(np.sqrt(w @ cov @ w) * np.sqrt(TRADING_DAYS))
    return ret, vol, (ret - rf) / vol if vol > 0 else np.nan


def random_portfolios(rets: pd.DataFrame, n: int = 5000, rf: float = 0.0, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mean, cov = rets.mean().values, rets.cov().values
    w = rng.dirichlet(np.ones(rets.shape[1]), size=n)
    port_ret = w @ mean * TRADING_DAYS
    port_vol = np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w)) * np.sqrt(TRADING_DAYS)
    out = pd.DataFrame({"Return": port_ret, "Volatility": port_vol})
    out["Sharpe"] = (out["Return"] - rf) / out["Volatility"]
    for i, col in enumerate(rets.columns):
        out[col] = w[:, i]
    return out


def optimise(rets: pd.DataFrame, objective: str = "max_sharpe", rf: float = 0.0,
             max_weight: float = 1.0) -> pd.Series:
    """Long-only weights that either maximise Sharpe or minimise volatility."""
    mean, cov = rets.mean().values, rets.cov().values
    k = rets.shape[1]
    bounds = [(0.0, max_weight)] * k
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)

    if objective == "min_vol":
        def fn(w):
            return _stats(w, mean, cov, rf)[1]
    else:
        def fn(w):
            s = _stats(w, mean, cov, rf)[2]
            return -s if np.isfinite(s) else 1e6

    res = minimize(fn, np.full(k, 1 / k), method="SLSQP", bounds=bounds, constraints=cons)
    w = np.clip(res.x if res.success else np.full(k, 1 / k), 0, None)
    return pd.Series(w / w.sum(), index=rets.columns)


def portfolio_summary(rets: pd.DataFrame, weights, rf: float = 0.0, confidence: float = 0.95) -> dict:
    port = portfolio_returns(rets, weights)
    equity = (1 + port).cumprod()
    return {
        "Annual return": annual_return(port),
        "Annual volatility": annual_vol(port),
        "Sharpe ratio": sharpe(port, rf),
        "Sortino ratio": sortino(port, rf),
        "Max drawdown": max_drawdown(equity),
        f"VaR {confidence:.0%} (1 day)": var_historical(port, confidence),
        f"CVaR {confidence:.0%} (1 day)": cvar_historical(port, confidence),
    }


# --------------------------------------------------------------------------- #
# volatility forecasting
# --------------------------------------------------------------------------- #
def ewma_vol_forecast(rets: pd.Series, lam: float = 0.94) -> float:
    """RiskMetrics EWMA one-day-ahead volatility (daily, as a fraction)."""
    r = rets.dropna().values
    var = np.var(r[:30]) if len(r) > 30 else np.var(r)
    for x in r:
        var = lam * var + (1 - lam) * x * x
    return float(np.sqrt(var))


def garch_forecast(rets: pd.Series, horizon: int = 10):
    """Real GARCH(1,1) via the `arch` package. Returns (daily vol forecast array,
    fitted params) or None if `arch` is not installed."""
    try:
        from arch import arch_model
    except ImportError:
        return None
    am = arch_model(rets.dropna() * 100, vol="GARCH", p=1, q=1, dist="t", rescale=False)
    res = am.fit(disp="off")
    fc = res.forecast(horizon=horizon, reindex=False)
    vol = np.sqrt(fc.variance.values[-1]) / 100
    return vol, res.params
