import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core import data, indicators as ind, risk, ui

ui.header("Strategy Backtest", "Test simple trading rules against buy-and-hold, including trading costs.")

c1, c2 = st.columns([1, 2])
with c1:
    symbol = ui.symbol_input("Ticker", "AAPL", key="bt_symbol")
with c2:
    start, end = ui.date_range(default_years=5, key="bt")

strategy = st.selectbox("Strategy", ["SMA crossover", "Price vs SMA", "RSI mean reversion", "MACD crossover",
                                     "Bollinger mean reversion"])
c1, c2, c3 = st.columns(3)
capital = c1.number_input("Starting capital ($)", 1000, 10_000_000, 10_000, step=1000)
cost_bps = c2.number_input("Cost per trade (bps)", 0.0, 100.0, 5.0, 1.0, help="Commission + spread. 10 bps = 0.10%.")
allow_short = c3.toggle("Allow short selling", False)

df = data.get_history(symbol, start, end)
if len(df) < 100:
    st.error("Need at least ~100 trading days.")
    st.stop()
close = df["Close"]

if strategy == "SMA crossover":
    a, b = st.columns(2)
    fast = a.number_input("Fast SMA", 5, 100, 50)
    slow = b.number_input("Slow SMA", 20, 300, 200)
    fast_ma, slow_ma = ind.sma(close, fast), ind.sma(close, slow)
    raw = np.where(fast_ma > slow_ma, 1, -1)
    raw = pd.Series(raw, index=close.index).where(slow_ma.notna(), 0)
    overlays = {f"SMA {fast}": fast_ma, f"SMA {slow}": slow_ma}
elif strategy == "Price vs SMA":
    n = st.number_input("SMA window", 5, 300, 20)
    ma = ind.sma(close, n)
    raw = pd.Series(np.where(close > ma, 1, -1), index=close.index).where(ma.notna(), 0)
    overlays = {f"SMA {n}": ma}
elif strategy == "RSI mean reversion":
    a, b = st.columns(2)
    lo = a.number_input("Buy below RSI", 5, 50, 30)
    hi = b.number_input("Exit above RSI", 50, 95, 55)
    r = ind.rsi(close)
    state, pos = 0, []
    for v in r:
        if state == 0 and v < lo:
            state = 1
        elif state == 1 and v > hi:
            state = 0
        pos.append(state)
    raw = pd.Series(pos, index=close.index)
    overlays = {}
elif strategy == "MACD crossover":
    m = ind.macd(close)
    raw = pd.Series(np.where(m["MACD"] > m["Signal"], 1, -1), index=close.index)
    overlays = {}
else:
    bb = ind.bollinger(close)
    state, pos = 0, []
    for c, lo_b, mid in zip(close, bb["Lower"], bb["Middle"]):
        if state == 0 and c < lo_b:
            state = 1
        elif state == 1 and c > mid:
            state = 0
        pos.append(state)
    raw = pd.Series(pos, index=close.index)
    overlays = {"BB middle": bb["Middle"], "BB lower": bb["Lower"]}

if not allow_short:
    raw = raw.clip(lower=0)

# Today's position comes from yesterday's signal: shift by one day so there's no look-ahead.
position = raw.shift(1).fillna(0)
asset_ret = close.pct_change().fillna(0)
turnover = position.diff().abs().fillna(position.abs())
strat_ret = position * asset_ret - turnover * cost_bps / 10_000
equity = capital * (1 + strat_ret).cumprod()
bh = capital * (1 + asset_ret).cumprod()

trades = position.diff().fillna(0)
entries = trades[trades > 0].index
exits = trades[trades < 0].index


def stats(eq: pd.Series, r: pd.Series) -> dict:
    return {"Final value": ui.money(eq.iloc[-1], 0), "Total return": ui.pct(eq.iloc[-1] / capital - 1),
            "CAGR": ui.pct(risk.cagr(eq)), "Volatility": ui.pct(risk.annual_vol(r)),
            "Sharpe": f"{risk.sharpe(r):.2f}", "Max drawdown": ui.pct(risk.max_drawdown(eq))}


s_strat, s_bh = stats(equity, strat_ret), stats(bh, asset_ret)
table = pd.DataFrame({"Strategy": s_strat, "Buy & hold": s_bh})
n_trades = int((turnover > 0).sum())
exposure = (position != 0).mean()

m = st.columns(4)
m[0].metric("Strategy return", s_strat["Total return"], f"vs B&H {s_bh['Total return']}", delta_color="off")
m[1].metric("Sharpe", s_strat["Sharpe"], f"B&H {s_bh['Sharpe']}", delta_color="off")
m[2].metric("Max drawdown", s_strat["Max drawdown"], f"B&H {s_bh['Max drawdown']}", delta_color="off")
m[3].metric("Trades / time in market", f"{n_trades} / {exposure:.0%}")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.04)
fig.add_trace(go.Scatter(x=close.index, y=close, name="Price", line=dict(color="#CBD5E1", width=1.3)), 1, 1)
for name, s in overlays.items():
    fig.add_trace(go.Scatter(x=s.index, y=s, name=name, line=dict(width=1)), 1, 1)
fig.add_trace(go.Scatter(x=entries, y=close.loc[entries], mode="markers", name="Buy",
                         marker=dict(symbol="triangle-up", size=10, color=ui.UP)), 1, 1)
fig.add_trace(go.Scatter(x=exits, y=close.loc[exits], mode="markers", name="Sell",
                         marker=dict(symbol="triangle-down", size=10, color=ui.DOWN)), 1, 1)
fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Strategy", line=dict(color=ui.ACCENT, width=2.5)), 2, 1)
fig.add_trace(go.Scatter(x=bh.index, y=bh, name="Buy & hold", line=dict(color="#F59E0B", width=1.5, dash="dot")), 2, 1)
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Equity ($)", row=2, col=1)
ui.chart(fig, 640, f"{strategy} on {symbol}")

st.dataframe(table, width="stretch")
st.caption("Each day's position uses only the signal from the previous close, so there is no look-ahead bias. "
           "Past performance on one stock says little about the future. Check several tickers and periods "
           "before trusting a rule.")
ui.data_status()
