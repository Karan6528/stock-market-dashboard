import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import data, risk, ui

ui.header("Risk Dashboard", "Single-stock risk against a benchmark: VaR, beta, Monte Carlo and a volatility forecast.")

c1, c2, c3 = st.columns(3)
with c1:
    symbol = ui.symbol_input("Ticker", "AAPL", key="rd_symbol")
with c2:
    bench = ui.symbol_input("Benchmark", "^GSPC", key="rd_bench")
confidence = c3.select_slider("Confidence level", [0.90, 0.95, 0.99], value=0.95, format_func=lambda x: f"{x:.0%}")
period = ui.period_picker("2Y", key="rd_period", options=["1Y", "2Y", "5Y", "10Y"])

df = data.get_history(symbol, period=period)
bm = data.get_history(bench, period=period)
if len(df) < 60:
    st.error("Not enough history.")
    st.stop()
rets = risk.returns(df["Close"])
bm_rets = risk.returns(bm["Close"])

var = risk.var_historical(rets, confidence)
cvar = risk.cvar_historical(rets, confidence)
m = st.columns(4)
m[0].metric(f"VaR {confidence:.0%} (1 day)", ui.pct(var), help="On 1 day in 20 (at 95%), the loss is expected to be at least this large.")
m[1].metric(f"CVaR {confidence:.0%} (1 day)", ui.pct(cvar), help="Average loss on those worst days.")
m[2].metric(f"Beta vs {bench}", f"{risk.beta(rets, bm_rets):.2f}")
m[3].metric("Max drawdown", ui.pct(risk.max_drawdown(df["Close"])))
m = st.columns(4)
m[0].metric("Annual return", ui.pct(risk.annual_return(rets)))
m[1].metric("Annual volatility", ui.pct(risk.annual_vol(rets)))
m[2].metric("Sharpe ratio (rf = 0)", f"{risk.sharpe(rets):.2f}")
m[3].metric("Sortino ratio", f"{risk.sortino(rets):.2f}")

tab_dist, tab_beta, tab_mc, tab_vol = st.tabs(["Return distribution", "Beta", "Monte Carlo", "Volatility forecast"])

with tab_dist:
    fig = go.Figure(go.Histogram(x=rets * 100, nbinsx=80, marker_color=ui.ACCENT, opacity=0.8, name="Daily returns"))
    fig.add_vline(x=var * 100, line_color="#F59E0B", line_dash="dash", annotation_text=f"VaR {var:.2%}")
    fig.add_vline(x=cvar * 100, line_color=ui.DOWN, line_dash="dash", annotation_text=f"CVaR {cvar:.2%}",
                  annotation_position="bottom left")
    fig.update_layout(xaxis_title="Daily return (%)", yaxis_title="Days", hovermode="closest")
    ui.chart(fig, 400)
    skew, kurt = rets.skew(), rets.kurt()
    st.caption(f"Skew {skew:.2f} · excess kurtosis {kurt:.2f}. Kurtosis above 0 means fat tails: extreme days "
               "happen more often than a normal distribution would suggest.")

with tab_beta:
    joined = pd.concat([rets.rename("asset"), bm_rets.rename("bench")], axis=1, join="inner").dropna()
    b = risk.beta(rets, bm_rets)
    alpha = (joined["asset"].mean() - b * joined["bench"].mean()) * 252
    x = np.linspace(joined["bench"].min(), joined["bench"].max(), 2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=joined["bench"] * 100, y=joined["asset"] * 100, mode="markers",
                             marker=dict(size=5, color=ui.ACCENT, opacity=0.5), name="Days"))
    fig.add_trace(go.Scatter(x=x * 100, y=(b * x + alpha / 252) * 100, mode="lines", name=f"β = {b:.2f}",
                             line=dict(color="#F59E0B", width=3)))
    fig.update_layout(xaxis_title=f"{bench} daily return (%)", yaxis_title=f"{symbol} daily return (%)",
                      hovermode="closest")
    ui.chart(fig, 420)
    r2 = joined.corr().iloc[0, 1] ** 2
    st.caption(f"Beta {b:.2f}: a 1% market move has historically come with a {b:.2f}% move in {symbol}. "
               f"Alpha {alpha:+.2%} p.a. · R² {r2:.2f} (share of variance explained by the market).")

with tab_mc:
    c1, c2 = st.columns(2)
    days = c1.slider("Horizon (trading days)", 10, 252, 60)
    n_paths = c2.select_slider("Simulations", [500, 1000, 2000, 5000], value=2000)
    last = float(df["Close"].iloc[-1])
    paths = risk.monte_carlo_paths(last, rets, days, n_paths)
    bands = risk.percentile_bands(paths)
    future = pd.bdate_range(df.index[-1], periods=days + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-120:], y=df["Close"].iloc[-120:], name="History", line=dict(color="white")))
    for i in range(min(40, n_paths)):
        fig.add_trace(go.Scatter(x=future, y=paths[:, i], mode="lines", line=dict(width=0.6, color="rgba(56,189,248,0.25)"),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future, y=bands["p95"], name="95th pct", line=dict(color=ui.UP, dash="dash")))
    fig.add_trace(go.Scatter(x=future, y=bands["p50"], name="Median", line=dict(color="#F59E0B", width=3)))
    fig.add_trace(go.Scatter(x=future, y=bands["p5"], name="5th pct", line=dict(color=ui.DOWN, dash="dash")))
    ui.chart(fig, 460, f"{n_paths:,} simulated paths (geometric Brownian motion)")
    end_px = paths[-1]
    c = st.columns(4)
    c[0].metric("Median price", ui.money(np.median(end_px)))
    c[1].metric("5th percentile", ui.money(np.percentile(end_px, 5)))
    c[2].metric("95th percentile", ui.money(np.percentile(end_px, 95)))
    c[3].metric("Chance of a loss", ui.pct((end_px < last).mean(), 1))

with tab_vol:
    ewma = risk.ewma_vol_forecast(rets)
    hist_vol = rets.tail(20).std()
    garch = risk.garch_forecast(rets, horizon=20)
    c = st.columns(3)
    c[0].metric("Realised vol (20d, ann.)", ui.pct(hist_vol * np.sqrt(252)))
    c[1].metric("EWMA forecast (ann.)", ui.pct(ewma * np.sqrt(252)), help="RiskMetrics, λ = 0.94")
    if garch is not None:
        vol_path, params = garch
        c[2].metric("GARCH(1,1) next day (ann.)", ui.pct(vol_path[0] * np.sqrt(252)))
        fig = go.Figure(go.Scatter(x=list(range(1, len(vol_path) + 1)), y=vol_path * np.sqrt(252) * 100,
                                   mode="lines+markers", line=dict(color=ui.ACCENT)))
        fig.add_hline(y=rets.std() * np.sqrt(252) * 100, line_dash="dot", annotation_text="long-run average")
        fig.update_layout(xaxis_title="Days ahead", yaxis_title="Annualised volatility (%)")
        ui.chart(fig, 340, "GARCH(1,1) volatility forecast: mean-reverts toward the long-run level")
        with st.expander("GARCH parameters"):
            st.dataframe(params.rename("value").to_frame(), width="stretch")
    else:
        c[2].metric("GARCH(1,1)", "n/a")
        st.caption("Install `arch` (`pip install arch`) to enable the GARCH(1,1) forecast.")
    roll = rets.rolling(20).std() * np.sqrt(252) * 100
    fig = go.Figure(go.Scatter(x=roll.index, y=roll, fill="tozeroy", line=dict(color="#A78BFA")))
    fig.update_layout(yaxis_title="Annualised volatility (%)")
    ui.chart(fig, 300, "Realised 20-day volatility")

ui.data_status()
