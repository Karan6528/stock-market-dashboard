"""Trade planning: position sizing, stop / target levels and a holding period
estimated from how this stock has actually behaved in the past."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind

STYLES = {
    # name: (ATR multiple for the stop, reward:risk, max holding days)
    "Short-term swing (days to weeks)": (2.0, 2.0, 30),
    "Medium-term (weeks to months)": (3.0, 2.5, 90),
    "Long-term (months)": (5.0, 3.0, 250),
}


@dataclass
class Plan:
    entry: float
    stop: float
    target: float
    atr: float
    risk_per_share: float
    shares: float
    invested: float
    cash_left: float
    max_loss: float
    potential_gain: float
    limited_by: str


def position_size(capital: float, price: float, atr: float, risk_pct: float, stop_atr: float,
                  reward_risk: float, fractional: bool = False) -> Plan:
    """Risk-based sizing: buy only as many shares as keeps the loss at the stop
    within `risk_pct` of capital, and never more than the cash available."""
    stop = max(price - stop_atr * atr, price * 0.01)
    risk_per_share = price - stop
    target = price + reward_risk * risk_per_share
    by_risk = capital * risk_pct / risk_per_share
    by_cash = capital / price
    raw = min(by_risk, by_cash)
    shares = raw if fractional else math.floor(raw)
    invested = shares * price
    return Plan(
        entry=price, stop=stop, target=target, atr=atr, risk_per_share=risk_per_share, shares=shares,
        invested=invested, cash_left=capital - invested, max_loss=shares * risk_per_share,
        potential_gain=shares * (target - price),
        limited_by="risk budget" if by_risk <= by_cash else "available cash",
    )


def historical_outcomes(df: pd.DataFrame, target_pct: float, stop_pct: float, max_days: int) -> pd.DataFrame:
    """Replay the same plan from every past day: enter at the close, then see
    whether the target (via the daily high) or the stop (via the daily low) was
    hit first within `max_days`. If both happen on the same day we assume the
    stop came first, which is the conservative choice."""
    c, h, lo = df["Close"].values, df["High"].values, df["Low"].values
    n = len(c)
    rows = []
    for i in range(n - 1):
        end = min(i + 1 + max_days, n)
        if end - (i + 1) < max_days:  # only complete windows
            break
        tgt, stp = c[i] * (1 + target_pct), c[i] * (1 - stop_pct)
        hw, lw = h[i + 1:end], lo[i + 1:end]
        t_hit = np.flatnonzero(hw >= tgt)
        s_hit = np.flatnonzero(lw <= stp)
        t_day = t_hit[0] + 1 if t_hit.size else None
        s_day = s_hit[0] + 1 if s_hit.size else None
        if t_day is not None and (s_day is None or t_day < s_day):
            rows.append((df.index[i], "Target", t_day, target_pct))
        elif s_day is not None:
            rows.append((df.index[i], "Stop", s_day, -stop_pct))
        else:
            rows.append((df.index[i], "Time limit", max_days, c[end - 1] / c[i] - 1))
    return pd.DataFrame(rows, columns=["Entry date", "Outcome", "Days", "Return"])


def summarise(outcomes: pd.DataFrame, stop_pct: float) -> dict:
    if outcomes.empty:
        return {}
    share = outcomes["Outcome"].value_counts(normalize=True)
    wins = outcomes[outcomes["Outcome"] == "Target"]["Days"]
    exp_r = outcomes["Return"].mean()
    return {
        "n": len(outcomes),
        "p_target": float(share.get("Target", 0.0)),
        "p_stop": float(share.get("Stop", 0.0)),
        "p_time": float(share.get("Time limit", 0.0)),
        "median_days_target": float(wins.median()) if len(wins) else float("nan"),
        "p80_days_target": float(wins.quantile(0.8)) if len(wins) else float("nan"),
        "median_days_all": float(outcomes["Days"].median()),
        "expected_return": float(exp_r),
        "expected_R": float(exp_r / stop_pct) if stop_pct else float("nan"),
    }


def setup_checks(df: pd.DataFrame, news_score: float | None = None) -> list[tuple[str, bool | None, str]]:
    """Simple, transparent rule checks: (name, passed, detail)."""
    c = df["Close"]
    last = c.iloc[-1]
    sma50, sma200 = ind.sma(c, 50).iloc[-1], ind.sma(c, 200).iloc[-1]
    rsi = ind.rsi(c).iloc[-1]
    mom = c.iloc[-1] / c.iloc[-min(63, len(c) - 1)] - 1
    checks = [
        ("Uptrend: price above 50-day average", bool(last > sma50) if pd.notna(sma50) else None,
         f"price {last:.2f} vs SMA50 {sma50:.2f}" if pd.notna(sma50) else "not enough data"),
        ("Long-term trend: 50-day above 200-day", bool(sma50 > sma200) if pd.notna(sma200) else None,
         f"SMA50 {sma50:.2f} vs SMA200 {sma200:.2f}" if pd.notna(sma200) else "needs 200 days of data"),
        ("Not overbought: RSI below 70", bool(rsi < 70) if pd.notna(rsi) else None, f"RSI {rsi:.0f}"),
        ("Positive 3-month momentum", bool(mom > 0), f"{mom:+.1%}"),
    ]
    if news_score is not None and np.isfinite(news_score):
        checks.append(("News tone not negative", bool(news_score > -0.05), f"average score {news_score:+.2f}"))
    return checks
