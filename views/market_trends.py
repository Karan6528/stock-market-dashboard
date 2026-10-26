import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import config, data, indicators as ind, ui

ui.header("Market & Sectors", "Major index trends and which sectors are leading or lagging.")

tab_idx, tab_sec = st.tabs(["Index trend", "Sector performance"])

with tab_idx:
    c1, c2 = st.columns([1, 2])
    name = c1.selectbox("Index", list(config.MARKET_INDICES))
    with c2:
        period = ui.period_picker("1Y", key="mt_period", options=["3M", "6M", "1Y", "2Y", "5Y"])
    sym = config.MARKET_INDICES[name]
    df = data.get_history(sym, period=period)
    close = df["Close"]
    sma50, sma200 = ind.sma(close, 50), ind.sma(close, 200)
    ema20 = ind.ema(close, 20)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=close, name=name, line=dict(width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=ema20, name="EMA 20", line=dict(width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=sma50, name="SMA 50", line=dict(width=1.3)))
    if sma200.notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=sma200, name="SMA 200", line=dict(width=1.3)))
    ui.chart(fig, 460, f"{name} ({sym})")

    last = close.iloc[-1]
    checks = {
        "Above 20-day EMA": last > ema20.iloc[-1],
        "Above 50-day SMA": last > sma50.iloc[-1] if pd.notna(sma50.iloc[-1]) else None,
        "50-day above 200-day (golden cross)": (sma50.iloc[-1] > sma200.iloc[-1])
        if pd.notna(sma200.iloc[-1]) else None,
    }
    valid = [v for v in checks.values() if v is not None]
    score = sum(valid)
    verdict = "Bullish" if score == len(valid) else "Bearish" if score == 0 else "Mixed"
    m = st.columns(4)
    m[0].metric("Trend", verdict)
    m[1].metric("Period return", ui.pct(last / close.iloc[0] - 1, sign=True))
    m[2].metric("From period high", ui.pct(last / close.max() - 1))
    m[3].metric("RSI (14)", f"{ind.rsi(close).iloc[-1]:.1f}")
    st.caption(" · ".join(f"{'✅' if v else '❌'} {k}" for k, v in checks.items() if v is not None))

with tab_sec:
    period = ui.period_picker("1M", key="sec_period", options=["1M", "3M", "6M", "1Y"])
    rows = []
    closes = {}
    for sector, etf in config.SECTOR_ETFS.items():
        h = data.get_history(etf, period=period)
        if len(h) > 1:
            rows.append({"Sector": sector, "ETF": etf, "Return": h["Close"].iloc[-1] / h["Close"].iloc[0] - 1})
            closes[sector] = h["Close"] / h["Close"].iloc[0] * 100
    if not rows:
        st.warning("No sector data available.")
        st.stop()
    perf = pd.DataFrame(rows).sort_values("Return")
    fig = go.Figure(go.Bar(
        x=perf["Return"] * 100, y=perf["Sector"], orientation="h",
        marker_color=[ui.UP if r >= 0 else ui.DOWN for r in perf["Return"]],
        text=[f"{r:+.1%}" for r in perf["Return"]], textposition="outside",
        customdata=perf["ETF"], hovertemplate="%{y} (%{customdata}): %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Return (%)", hovermode="closest")
    ui.chart(fig, 460, f"Sector ETF returns · {period}")

    best, worst = perf.iloc[-1], perf.iloc[0]
    c1, c2 = st.columns(2)
    c1.success(f"Leader: **{best['Sector']}** ({best['ETF']}) {best['Return']:+.2%}")
    c2.warning(f"Laggard: **{worst['Sector']}** ({worst['ETF']}) {worst['Return']:+.2%}")

    norm = pd.DataFrame(closes).dropna()
    fig2 = px.line(norm, labels={"value": "Growth of 100", "index": "", "variable": "Sector"})
    ui.chart(fig2, 420, "Relative performance (rebased to 100)")

ui.data_status()
