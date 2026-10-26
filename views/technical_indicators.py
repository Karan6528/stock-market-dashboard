import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core import data, indicators as ind, ui

ui.header("Technical Indicators", "Computed locally from price data: no API calls or rate limits, and it works offline.")

with st.sidebar:
    st.subheader("Indicator settings")
    overlays = st.multiselect("Price overlays", ["SMA", "EMA", "Bollinger Bands"], default=["SMA", "Bollinger Bands"])
    sma_win = st.number_input("SMA window", 5, 250, 50)
    ema_win = st.number_input("EMA span", 5, 250, 20)
    panels = st.multiselect("Lower panels", ["RSI", "MACD", "Stochastic", "ATR", "ADX", "CCI", "OBV"],
                            default=["RSI", "MACD"])

symbols = ui.symbols_input("Tickers (comma-separated)", "AAPL", key="ti_syms")
start, end = ui.date_range(default_years=1, key="ti")

for symbol in symbols[:5]:
    df = data.get_history(symbol, start, end)
    if len(df) < 30:
        st.warning(f"Not enough data for {symbol}.")
        continue
    close = df["Close"]
    n_rows = 1 + len(panels)
    heights = [0.5] + [0.5 / max(len(panels), 1)] * len(panels) if panels else [1]
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=heights, subplot_titles=[symbol] + panels)
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=close,
                                 name="Price", increasing_line_color=ui.UP, decreasing_line_color=ui.DOWN), 1, 1)
    if "SMA" in overlays:
        fig.add_trace(go.Scatter(x=df.index, y=ind.sma(close, sma_win), name=f"SMA {sma_win}",
                                 line=dict(color="#F59E0B", width=1.4)), 1, 1)
    if "EMA" in overlays:
        fig.add_trace(go.Scatter(x=df.index, y=ind.ema(close, ema_win), name=f"EMA {ema_win}",
                                 line=dict(color="#A78BFA", width=1.4)), 1, 1)
    if "Bollinger Bands" in overlays:
        bb = ind.bollinger(close)
        fig.add_trace(go.Scatter(x=df.index, y=bb["Upper"], name="BB upper",
                                 line=dict(color="#64748B", width=1, dash="dot")), 1, 1)
        fig.add_trace(go.Scatter(x=df.index, y=bb["Lower"], name="BB lower", fill="tonexty",
                                 fillcolor="rgba(100,116,139,0.08)",
                                 line=dict(color="#64748B", width=1, dash="dot")), 1, 1)

    for i, panel in enumerate(panels, start=2):
        if panel == "RSI":
            fig.add_trace(go.Scatter(x=df.index, y=ind.rsi(close), name="RSI", line=dict(color=ui.ACCENT)), i, 1)
            fig.add_hline(y=70, line_dash="dot", line_color=ui.DOWN, row=i, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color=ui.UP, row=i, col=1)
            fig.update_yaxes(range=[0, 100], row=i, col=1)
        elif panel == "MACD":
            m = ind.macd(close)
            fig.add_trace(go.Bar(x=df.index, y=m["Histogram"], name="MACD hist",
                                 marker_color=[ui.UP if v >= 0 else ui.DOWN for v in m["Histogram"].fillna(0)]), i, 1)
            fig.add_trace(go.Scatter(x=df.index, y=m["MACD"], name="MACD", line=dict(color=ui.ACCENT)), i, 1)
            fig.add_trace(go.Scatter(x=df.index, y=m["Signal"], name="Signal", line=dict(color="#F59E0B")), i, 1)
        elif panel == "Stochastic":
            s = ind.stochastic(df)
            fig.add_trace(go.Scatter(x=df.index, y=s["%K"], name="%K", line=dict(color=ui.ACCENT)), i, 1)
            fig.add_trace(go.Scatter(x=df.index, y=s["%D"], name="%D", line=dict(color="#F59E0B")), i, 1)
            fig.add_hline(y=80, line_dash="dot", line_color=ui.DOWN, row=i, col=1)
            fig.add_hline(y=20, line_dash="dot", line_color=ui.UP, row=i, col=1)
        elif panel == "ATR":
            fig.add_trace(go.Scatter(x=df.index, y=ind.atr(df), name="ATR", line=dict(color="#F472B6")), i, 1)
        elif panel == "ADX":
            fig.add_trace(go.Scatter(x=df.index, y=ind.adx(df), name="ADX", line=dict(color="#2DD4BF")), i, 1)
            fig.add_hline(y=25, line_dash="dot", line_color="#94A3B8", row=i, col=1)
        elif panel == "CCI":
            fig.add_trace(go.Scatter(x=df.index, y=ind.cci(df), name="CCI", line=dict(color="#FACC15")), i, 1)
            fig.add_hline(y=100, line_dash="dot", line_color=ui.DOWN, row=i, col=1)
            fig.add_hline(y=-100, line_dash="dot", line_color=ui.UP, row=i, col=1)
        elif panel == "OBV":
            fig.add_trace(go.Scatter(x=df.index, y=ind.obv(df), name="OBV", line=dict(color="#A78BFA")), i, 1)

    fig.update_layout(xaxis_rangeslider_visible=False, showlegend=False)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    ui.chart(fig, 380 + 170 * len(panels))

    rsi_now = ind.rsi(close).iloc[-1]
    m = ind.macd(close).iloc[-1]
    sma_now = ind.sma(close, sma_win).iloc[-1]
    signals = [
        f"RSI {rsi_now:.0f}: " + ("overbought" if rsi_now > 70 else "oversold" if rsi_now < 30 else "neutral"),
        "MACD " + ("above" if m["MACD"] > m["Signal"] else "below") + " signal line",
        f"Price {'above' if close.iloc[-1] > sma_now else 'below'} SMA {sma_win}",
    ]
    st.caption(" · ".join(signals))
    st.divider()

with st.expander("What do these indicators mean?"):
    st.markdown("""
- **SMA / EMA**: simple and exponentially weighted moving averages. They smooth price to show trend.
- **Bollinger Bands**: SMA 20 ± 2 standard deviations. Price outside the bands is statistically stretched.
- **RSI (Wilder, 14)**: momentum on a 0–100 scale. Above 70 is often called overbought, below 30 oversold.
- **MACD**: EMA 12 − EMA 26, with a 9-day signal line. Crossovers indicate momentum shifts.
- **Stochastic**: where the close sits within the recent high–low range.
- **ATR**: average true range, a measure of volatility in price units.
- **ADX**: trend strength regardless of direction. Above 25 is usually considered a trend.
- **CCI**: deviation of typical price from its average.
- **OBV**: cumulative volume flow, used to confirm price moves.
""")
ui.data_status()
