import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import data, risk, ui

ui.header("Compare & Correlate", "Rebased performance, risk/return and the correlation of daily returns.")

symbols = ui.symbols_input("Tickers (comma-separated)", "AAPL, MSFT, GOOGL, NVDA, TSLA", key="cmp_syms", min_count=2)
start, end = ui.date_range(default_years=2, key="cmp")

closes = data.get_closes(symbols, start, end)
if closes.shape[1] < 2 or len(closes) < 20:
    st.error("Need at least two tickers with overlapping history.")
    st.stop()
rets = risk.returns(closes)

tab_perf, tab_corr, tab_rr = st.tabs(["Performance", "Correlation", "Risk vs return"])

with tab_perf:
    rebased = closes / closes.iloc[0] * 100
    fig = px.line(rebased, labels={"value": "Growth of 100", "Date": "", "variable": "Ticker"})
    ui.chart(fig, 460, "Rebased to 100 at start date")
    summary = (closes.iloc[-1] / closes.iloc[0] - 1).sort_values(ascending=False)
    cols = st.columns(min(len(summary), 6))
    for col, (sym, r) in zip(cols, summary.items()):
        col.metric(sym, ui.pct(r, sign=True))

with tab_corr:
    method = st.radio("Method", ["pearson", "spearman"], horizontal=True,
                      help="Spearman is rank-based and less sensitive to outliers.")
    corr = rets.corr(method=method)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(hovermode="closest")
    ui.chart(fig, 480, "Correlation of daily returns")
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    if len(upper):
        hi, lo = upper.idxmax(), upper.idxmin()
        st.caption(f"Most correlated: **{hi[0]} / {hi[1]}** ({upper.max():.2f}) · "
                   f"least correlated: **{lo[0]} / {lo[1]}** ({upper.min():.2f}). "
                   "Lower correlation means more diversification.")
    window = st.slider("Rolling window (days)", 20, 250, 60)
    a, b = st.columns(2)
    s1 = a.selectbox("Pair: first", list(closes.columns), index=0)
    s2 = b.selectbox("Pair: second", list(closes.columns), index=1)
    roll = rets[s1].rolling(window).corr(rets[s2])
    fig2 = go.Figure(go.Scatter(x=roll.index, y=roll, mode="lines", line=dict(color=ui.ACCENT)))
    fig2.add_hline(y=0, line_color="#64748B")
    fig2.update_layout(yaxis_range=[-1, 1], yaxis_title="Correlation")
    ui.chart(fig2, 320, f"Rolling {window}-day correlation: {s1} vs {s2}")

with tab_rr:
    stats = []
    for s in closes.columns:
        stats.append({"Ticker": s, "Annual return": risk.annual_return(rets[s]),
                      "Annual volatility": risk.annual_vol(rets[s]), "Sharpe": risk.sharpe(rets[s]),
                      "Max drawdown": risk.max_drawdown(closes[s])})
    import pandas as pd
    stats = pd.DataFrame(stats)
    fig = go.Figure(go.Scatter(
        x=stats["Annual volatility"] * 100, y=stats["Annual return"] * 100, mode="markers+text",
        text=stats["Ticker"], textposition="top center", marker=dict(size=16, color=stats["Sharpe"],
                                                                     colorscale="Viridis", showscale=True,
                                                                     colorbar=dict(title="Sharpe")),
    ))
    fig.update_layout(xaxis_title="Volatility (% p.a.)", yaxis_title="Return (% p.a.)", hovermode="closest")
    ui.chart(fig, 440, "Risk vs return")
    st.dataframe(stats, hide_index=True, width="stretch", column_config={
        "Annual return": st.column_config.NumberColumn(format="percent"),
        "Annual volatility": st.column_config.NumberColumn(format="percent"),
        "Sharpe": st.column_config.NumberColumn(format="%.2f"),
        "Max drawdown": st.column_config.NumberColumn(format="percent"),
    })

ui.data_status()
