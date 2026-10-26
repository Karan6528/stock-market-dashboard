import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression

from core import data, risk, ui

ui.header("Long-Term Projection",
          "What an investment could be worth in 1–10 years, shown as a range of outcomes rather than a single number.")

symbols = ui.symbols_input("Tickers", "AAPL, MSFT, GOOGL", key="ltp_syms")
c1, c2, c3 = st.columns(3)
years = c1.selectbox("Horizon (years)", [1, 2, 3, 5, 10], index=1)
lookback = c2.selectbox("Estimate from history", ["3Y", "5Y", "10Y"], index=1)
n_paths = c3.select_slider("Simulations", [1000, 2000, 5000], value=2000)

st.markdown("**Investment per ticker ($)**")
inv_cols = st.columns(min(len(symbols), 6))
investments = {s: inv_cols[i % len(inv_cols)].number_input(s, 0.0, 1e8, 1000.0, 100.0, key=f"inv_{s}")
               for i, s in enumerate(symbols)}

days = int(years * 252)
total_paths = np.zeros((days + 1, n_paths))
rows = []
first = None
for i, s in enumerate(symbols):
    df = data.get_history(s, period=lookback)
    if len(df) < 250:
        st.warning(f"Skipping {s}: not enough history.")
        continue
    close = df["Close"]
    rets = risk.returns(close)
    paths = risk.monte_carlo_paths(1.0, rets, days, n_paths, seed=100 + i)  # growth multiples
    total_paths += paths * investments[s]
    end = paths[-1]
    rows.append({"Ticker": s, "Invested": investments[s], "Hist. CAGR": risk.cagr(close),
                 "Hist. volatility": risk.annual_vol(rets),
                 "Pessimistic (5%)": investments[s] * np.percentile(end, 5),
                 "Median": investments[s] * np.median(end),
                 "Optimistic (95%)": investments[s] * np.percentile(end, 95),
                 "Chance of loss": (end < 1).mean()})

    if first is None:  # trend-line comparison uses the first usable ticker
        first = (s, df)

if not rows:
    st.stop()
table = pd.DataFrame(rows).set_index("Ticker")
invested = table["Invested"].sum()
total_end = total_paths[-1]

st.subheader(f"Portfolio after {years} year{'s' if years > 1 else ''}")
m = st.columns(4)
m[0].metric("Invested", ui.money(invested, 0))
m[1].metric("Median outcome", ui.money(np.median(total_end), 0), ui.pct(np.median(total_end) / invested - 1, 1, sign=True))
m[2].metric("Range (5%–95%)", f"{ui.money(np.percentile(total_end, 5), 0)} – {ui.money(np.percentile(total_end, 95), 0)}")
m[3].metric("Chance of a loss", ui.pct((total_end < invested).mean(), 1))

bands = risk.percentile_bands(total_paths)
x = pd.bdate_range(pd.Timestamp.today().normalize(), periods=days + 1)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=bands["p95"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=x, y=bands["p5"], fill="tonexty", fillcolor="rgba(56,189,248,0.12)", line=dict(width=0),
                         name="5–95% range"))
fig.add_trace(go.Scatter(x=x, y=bands["p75"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=x, y=bands["p25"], fill="tonexty", fillcolor="rgba(56,189,248,0.28)", line=dict(width=0),
                         name="25–75% range"))
fig.add_trace(go.Scatter(x=x, y=bands["p50"], name="Median", line=dict(color=ui.ACCENT, width=3)))
fig.add_hline(y=invested, line_dash="dot", line_color="#94A3B8", annotation_text="invested")
fig.update_layout(yaxis_title="Portfolio value ($)")
ui.chart(fig, 440, "Fan chart of simulated outcomes")

st.dataframe(table, width="stretch", column_config={
    "Invested": st.column_config.NumberColumn(format="$%.0f"),
    "Hist. CAGR": st.column_config.NumberColumn(format="percent"),
    "Hist. volatility": st.column_config.NumberColumn(format="percent"),
    "Pessimistic (5%)": st.column_config.NumberColumn(format="$%.0f"),
    "Median": st.column_config.NumberColumn(format="$%.0f"),
    "Optimistic (95%)": st.column_config.NumberColumn(format="$%.0f"),
    "Chance of loss": st.column_config.NumberColumn(format="percent"),
})
st.caption("Simulation: geometric Brownian motion using each stock's historical mean and volatility of log returns. "
           "Each stock is simulated independently, so the range for the combined portfolio is somewhat too narrow "
           "for correlated holdings. The method also assumes the past return rate continues.")

with st.expander("Why not just fit a trend line? (comparison with the original approach)"):
    s, df = first
    t = np.arange(len(df)).reshape(-1, 1)
    lr = LinearRegression().fit(t, df["Close"].values)
    future_t = np.arange(len(df), len(df) + days).reshape(-1, 1)
    trend = lr.predict(future_t)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(300):  # bootstrap the trend for a confidence band
        idx = rng.integers(0, len(df), len(df))
        boots.append(LinearRegression().fit(t[idx], df["Close"].values[idx]).predict(future_t))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    fut_idx = pd.bdate_range(df.index[-1], periods=days + 1)[1:]
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=df.index, y=df["Close"], name=s, line=dict(color="white")))
    f2.add_trace(go.Scatter(x=fut_idx, y=hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    f2.add_trace(go.Scatter(x=fut_idx, y=lo, fill="tonexty", fillcolor="rgba(245,158,11,0.2)", line=dict(width=0),
                            name="95% CI of the trend line"))
    f2.add_trace(go.Scatter(x=fut_idx, y=trend, name="Linear trend", line=dict(color="#F59E0B", dash="dash")))
    ui.chart(f2, 380)
    st.markdown(
        "The trend line's confidence band is narrow because it only measures uncertainty in the **slope**, "
        "not in where the price actually goes. A high R² on the training data doesn't make it a good forecast. "
        "The Monte Carlo fan above is a more honest picture of the range of outcomes."
    )
ui.data_status()
ui.disclaimer()
