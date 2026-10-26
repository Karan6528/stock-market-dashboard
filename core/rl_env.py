"""A small, honest trading environment for reinforcement learning (Gymnasium API).

Changes vs the original gym version:
- observations are scale-free (returns, indicators, position), not raw prices
  and cash, so the agent can generalise across price levels and tickers;
- reward is the daily log change in portfolio value (not cumulative P/L,
  which the original added up again every step);
- a cost per trade, and an all-in / all-out position instead of 1 share.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # optional dependency
    gym = None

from . import indicators as ind

WINDOW = 10


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    ret = close.pct_change()
    f = pd.DataFrame(index=df.index)
    for i in range(WINDOW):
        f[f"r{i}"] = ret.shift(i) * 10
    f["sma20"] = (close / ind.sma(close, 20) - 1) * 5
    f["sma50"] = (close / ind.sma(close, 50) - 1) * 5
    f["rsi"] = ind.rsi(close) / 100 - 0.5
    f["vol20"] = ret.rolling(20).std() * 20
    return f.dropna()


if gym is not None:

    class TradingEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, df: pd.DataFrame, cost_bps: float = 5.0, initial_cash: float = 10_000.0):
            super().__init__()
            feats = make_features(df)
            self.prices = df["Close"].reindex(feats.index).values.astype(float)
            self.dates = feats.index
            self.X = feats.values.astype(np.float32)
            self.cost = cost_bps / 10_000
            self.initial_cash = initial_cash
            self.action_space = spaces.Discrete(3)  # 0 = flat/sell, 1 = hold, 2 = long/buy
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.X.shape[1] + 1,), dtype=np.float32)

        def _obs(self):
            return np.append(self.X[self.t], np.float32(self.position)).astype(np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.t = 0
            self.position = 0  # 0 = cash, 1 = fully invested
            self.value = self.initial_cash
            self.history = [(self.dates[0], self.value, self.position, "START", self.prices[0])]
            return self._obs(), {}

        def step(self, action):
            action = int(action)
            new_pos = 1 if action == 2 else 0 if action == 0 else self.position
            trade = new_pos != self.position
            if trade:
                self.value *= 1 - self.cost
            self.position = new_pos
            self.t += 1
            ret = self.prices[self.t] / self.prices[self.t - 1] - 1
            old = self.value
            self.value *= 1 + self.position * ret
            reward = float(np.log(self.value / old)) if old > 0 else 0.0
            label = ("BUY" if new_pos else "SELL") if trade else "HOLD"
            self.history.append((self.dates[self.t], self.value, self.position, label, self.prices[self.t]))
            terminated = self.t >= len(self.prices) - 1
            return self._obs(), reward, terminated, False, {}

        def history_frame(self) -> pd.DataFrame:
            return pd.DataFrame(self.history, columns=["Date", "Value", "Position", "Action", "Price"]).set_index("Date")
else:  # pragma: no cover
    TradingEnv = None
