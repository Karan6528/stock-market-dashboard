import datetime as dt

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core import data, risk, ui

ui.header("Price Charts", "Candlestick, OHLC, line and area charts, plus the return since any start date.")

c1, c2, c3 = st.columns([1.2, 1, 1.4])
with c1:
    symbol = ui.symbol_input("Ticker", "AAPL", key="pc_symbol")
style = c2.selectbox("Chart type", ["Candlestick", "OHLC", "Line", "Area"])
with c3:
    start, end = ui.date_range(default_years=1, key="pc")

df = data.get_history(symbol, start, end)
if df.empty:
    st.error(f"No data for {symbol} in this range.")
    st.stop()

log_scale = st.toggle("Log scale", value=False)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.03)
if style == "Candlestick":
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                 name=symbol, increasing_line_color=ui.UP, decreasing_line_color=ui.DOWN), 1, 1)
elif style == "OHLC":
    fig.add_trace(go.Ohlc(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                          name=symbol, increasing_line_color=ui.UP, decreasing_line_color=ui.DOWN), 1, 1)
else:
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name=symbol, mode="lines",
                             fill="tozeroy" if style == "Area" else None, line=dict(width=2)), 1, 1)
colors = [ui.UP if c >= o else ui.DOWN for o, c in zip(df["Open"], df["Close"])]
fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors, name="Volume", opacity=0.6), 2, 1)
fig.update_layout(xaxis_rangeslider_visible=False, showlegend=False)
fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
if log_scale:
    fig.update_yaxes(type="log", row=1, col=1)
ui.chart(fig, 560, f"{symbol} · {start:%d %b %Y} to {end:%d %b %Y}")

close = df["Close"]
rets = risk.returns(close)
m = st.columns(5)
m[0].metric("Total return", ui.pct(close.iloc[-1] / close.iloc[0] - 1, sign=True),
            f"{close.iloc[-1] - close.iloc[0]:+.2f} $")
m[1].metric("CAGR", ui.pct(risk.cagr(close)))
m[2].metric("Volatility (ann.)", ui.pct(risk.annual_vol(rets)))
m[3].metric("Max drawdown", ui.pct(risk.max_drawdown(close)))
m[4].metric("Best / worst day", f"{rets.max():+.1%} / {rets.min():+.1%}")

st.subheader("What if you had invested?")
c1, c2 = st.columns(2)
amount = c1.number_input("Amount invested ($)", 100.0, 10_000_000.0, 1_000.0, step=100.0)
year = c2.selectbox("Starting year", list(range(dt.date.today().year - 1, 1999, -1)), index=9)
long = data.get_history(symbol, start=dt.date(year, 1, 1), end=dt.date.today())
if len(long) > 1:
    growth = long["Close"] / long["Close"].iloc[0] * amount
    st.metric(f"Value today of {ui.money(amount, 0)} invested in {symbol} on {long.index[0]:%d %b %Y}",
              ui.money(growth.iloc[-1]), ui.pct(growth.iloc[-1] / amount - 1, sign=True))
    g = go.Figure(go.Scatter(x=growth.index, y=growth, mode="lines", fill="tozeroy", line=dict(color=ui.ACCENT)))
    g.update_layout(yaxis_title="Value ($)")
    ui.chart(g, 320)
    st.caption("Prices are split- and dividend-adjusted, so this approximates a total return with dividends reinvested.")

with st.expander("Raw data"):
    st.dataframe(df.sort_index(ascending=False), width="stretch")
    ui.to_csv_download(df, f"{symbol}_prices.csv")
ui.data_status()
