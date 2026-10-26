import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import data, risk, ui

ui.header("Portfolio Analytics", "Risk metrics, efficient frontier and rolling volatility for a weighted portfolio.")

c1, c2 = st.columns([2, 1])
with c1:
    symbols = ui.symbols_input("Tickers", "AAPL, MSFT, GOOGL, TSLA, AMZN, META", key="pa_syms", min_count=2)
with c2:
    rf = st.number_input("Risk-free rate (% p.a.)", 0.0, 10.0, 3.0, 0.25) / 100
start, end = ui.date_range(default_years=3, key="pa")

st.markdown("**Weights**")
default_w = pd.DataFrame({"Ticker": symbols, "Weight %": [round(100 / len(symbols), 2)] * len(symbols)})
w_df = st.data_editor(default_w, hide_index=True, width="content", disabled=["Ticker"],
                      key=f"weights_{'_'.join(symbols)}",
                      column_config={"Weight %": st.column_config.NumberColumn(min_value=0, max_value=100, step=1)})
total = w_df["Weight %"].sum()
if total <= 0:
    st.error("Weights must add up to more than 0.")
    st.stop()
if abs(total - 100) > 0.01:
    st.caption(f"Weights add up to {total:.1f}%, so they've been rescaled to 100%.")
weights = pd.Series(w_df["Weight %"].values / total, index=symbols)

closes = data.get_closes(symbols, start, end)
missing = [s for s in symbols if s not in closes.columns]
if missing or len(closes) < 60:
    st.error(f"Not enough overlapping history{' for ' + ', '.join(missing) if missing else ''}.")
    st.stop()
rets = risk.returns(closes)
weights = weights[closes.columns]
port = risk.portfolio_returns(rets, weights.values)
equity = (1 + port).cumprod()

stats = risk.portfolio_summary(rets, weights.values, rf)
cols = st.columns(len(stats))
for col, (k, v) in zip(cols, stats.items()):
    col.metric(k, f"{v:.2f}" if "ratio" in k else ui.pct(v))

tab_growth, tab_frontier, tab_vol, tab_corr = st.tabs(["Growth & drawdown", "Efficient frontier",
                                                        "Volatility", "Diversification"])

with tab_growth:
    growth = pd.concat([equity.rename("Portfolio"), (closes / closes.iloc[0])], axis=1)
    fig = px.line(growth * 100, labels={"value": "Growth of 100", "Date": "", "variable": ""})
    fig.update_traces(line=dict(width=1), opacity=0.55)
    fig.update_traces(selector=dict(name="Portfolio"), line=dict(width=3, color="white"), opacity=1)
    ui.chart(fig, 420, "Portfolio vs individual holdings")
    dd = risk.drawdown(equity)
    fig2 = go.Figure(go.Scatter(x=dd.index, y=dd * 100, fill="tozeroy", line=dict(color=ui.DOWN)))
    fig2.update_layout(yaxis_title="Drawdown (%)")
    ui.chart(fig2, 260, "Underwater chart (drawdown from previous peak)")

with tab_frontier:
    n_sims = st.select_slider("Random portfolios", [1000, 2500, 5000, 10000], value=5000)
    sims = risk.random_portfolios(rets, n_sims, rf)
    w_sharpe = risk.optimise(rets, "max_sharpe", rf)
    w_minvol = risk.optimise(rets, "min_vol", rf)

    def point(w):
        p = risk.portfolio_returns(rets, w.values)
        return risk.annual_vol(p), risk.annual_return(p)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=sims["Volatility"] * 100, y=sims["Return"] * 100, mode="markers",
                               marker=dict(size=4, color=sims["Sharpe"], colorscale="Viridis", showscale=True,
                                           colorbar=dict(title="Sharpe")), name="Random", hoverinfo="skip"))
    for name, w, sym, color in [("Max Sharpe", w_sharpe, "star", "#FACC15"), ("Min volatility", w_minvol, "diamond", ui.ACCENT),
                                ("Your weights", weights, "circle", "white")]:
        v, r = point(w)
        fig.add_trace(go.Scatter(x=[v * 100], y=[r * 100], mode="markers", name=name,
                                 marker=dict(symbol=sym, size=16, color=color, line=dict(width=1, color="black"))))
    fig.update_layout(xaxis_title="Volatility (% p.a.)", yaxis_title="Expected return (% p.a.)", hovermode="closest")
    ui.chart(fig, 480, "Efficient frontier")
    table = pd.DataFrame({"Your weights": weights, "Max Sharpe": w_sharpe, "Min volatility": w_minvol})
    st.dataframe(table, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="percent") for c in table.columns})
    st.caption("Optimised weights are based on past returns and are very sensitive to the date range. "
               "Treat them as an illustration of mean-variance theory, not a recommendation.")

with tab_vol:
    window = st.slider("Rolling window (days)", 10, 120, 30)
    roll = rets.rolling(window).std() * np.sqrt(252) * 100
    roll["Portfolio"] = port.rolling(window).std() * np.sqrt(252) * 100
    fig = px.line(roll, labels={"value": "Annualised volatility (%)", "Date": "", "variable": ""})
    fig.update_traces(selector=dict(name="Portfolio"), line=dict(width=3, color="white"))
    ui.chart(fig, 420, f"Rolling {window}-day volatility")
    vol_tbl = pd.DataFrame({
        "Daily volatility": rets.std(),
        "Annual volatility": rets.std() * np.sqrt(252),
        "VaR 95% (1 day)": rets.quantile(0.05),
        "Risk contribution": weights * (rets.cov().values @ weights.values) / (weights.values @ rets.cov().values @ weights.values),
    })
    st.dataframe(vol_tbl, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="percent") for c in vol_tbl.columns})
    st.caption("Risk contribution shows each holding's share of total portfolio variance. A small weight in a "
               "volatile stock can still dominate the portfolio's risk.")

with tab_corr:
    corr = rets.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(hovermode="closest")
    ui.chart(fig, 460, "Correlation of daily returns")
    avg_corr = corr.where(~np.eye(len(corr), dtype=bool)).stack().mean()
    div_ratio = (weights * rets.std()).sum() / port.std()
    c1, c2 = st.columns(2)
    c1.metric("Average pairwise correlation", f"{avg_corr:.2f}")
    c2.metric("Diversification ratio", f"{div_ratio:.2f}",
              help="Weighted average of individual volatilities ÷ portfolio volatility. Higher = more diversification.")

ui.data_status()
