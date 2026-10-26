import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import data, portfolio_db as db, risk, ui

ui.header("My Portfolio", "Track your holdings and live profit/loss. Saved to data/portfolio.db.")

with st.sidebar:
    st.subheader(":material/add_circle: Add a holding")
    with st.form("add_holding", clear_on_submit=True):
        ticker = st.text_input("Ticker", "AAPL")
        shares = st.number_input("Shares", min_value=0.0001, value=10.0, step=1.0, format="%.4f")
        buy_price = st.number_input("Buy price ($)", min_value=0.0001, value=150.0, step=1.0)
        buy_date = st.date_input("Purchase date", dt.date.today(), max_value=dt.date.today())
        if st.form_submit_button("Add to portfolio", type="primary", width="stretch"):
            db.add(ticker, shares, buy_price, buy_date.isoformat())
            st.toast(f"Added {shares:g} × {ticker.upper()}", icon=":material/check:")

holdings = db.load()
if holdings.empty:
    st.info("Your portfolio is empty. Add holdings from the sidebar.", icon=":material/info:")
    if st.button("Load an example portfolio"):
        today = dt.date.today()
        for t, sh, px_, d in [("AAPL", 15, 150.0, 400), ("MSFT", 8, 310.0, 300), ("NVDA", 20, 45.0, 500),
                              ("JPM", 10, 140.0, 250)]:
            db.add(t, sh, px_, (today - dt.timedelta(days=d)).isoformat())
        st.rerun()
    st.stop()

prices = {t: data.get_quote(t).get("price", np.nan) for t in holdings["ticker"].unique()}
h = holdings.copy()
h["price"] = h["ticker"].map(prices)
h["cost"] = h["shares"] * h["buy_price"]
h["value"] = h["shares"] * h["price"]
h["pnl"] = h["value"] - h["cost"]
h["return"] = h["pnl"] / h["cost"]

total_cost, total_value = h["cost"].sum(), h["value"].sum()
day_change = 0.0
for t in h["ticker"].unique():
    q = data.get_quote(t)
    day_change += q.get("change", 0) * h.loc[h["ticker"] == t, "shares"].sum()

m = st.columns(4)
m[0].metric("Market value", ui.money(total_value))
m[1].metric("Total cost", ui.money(total_cost))
m[2].metric("Unrealised P/L", ui.money(total_value - total_cost), ui.pct(total_value / total_cost - 1, sign=True))
m[3].metric("Today", ui.money(day_change), ui.pct(day_change / (total_value - day_change), sign=True)
            if total_value else None)

st.subheader("Holdings")
st.caption("Edit cells directly, or select rows and press Delete, then save.")
edited = st.data_editor(
    holdings.drop(columns=["id"]), num_rows="dynamic", width="stretch", hide_index=True,
    key="holdings_editor",
    column_config={
        "ticker": st.column_config.TextColumn("Ticker", required=True),
        "shares": st.column_config.NumberColumn("Shares", min_value=0, format="%.4g"),
        "buy_price": st.column_config.NumberColumn("Buy price", min_value=0, format="$%.2f"),
        "buy_date": st.column_config.TextColumn("Purchase date (YYYY-MM-DD)"),
    },
)
if st.button("Save changes", icon=":material/save:"):
    db.replace_all(edited.dropna(subset=["ticker", "shares", "buy_price"]))
    st.toast("Portfolio saved", icon=":material/check:")
    st.rerun()

by_ticker = h.groupby("ticker").agg(shares=("shares", "sum"), cost=("cost", "sum"), value=("value", "sum"),
                                    price=("price", "last"))
by_ticker["avg cost"] = by_ticker["cost"] / by_ticker["shares"]
by_ticker["P/L"] = by_ticker["value"] - by_ticker["cost"]
by_ticker["return"] = by_ticker["P/L"] / by_ticker["cost"]
by_ticker["weight"] = by_ticker["value"] / by_ticker["value"].sum()
st.dataframe(by_ticker.sort_values("value", ascending=False), width="stretch", column_config={
    "shares": st.column_config.NumberColumn(format="%.4g"),
    "cost": st.column_config.NumberColumn(format="$%.2f"),
    "value": st.column_config.NumberColumn(format="$%.2f"),
    "price": st.column_config.NumberColumn(format="$%.2f"),
    "avg cost": st.column_config.NumberColumn(format="$%.2f"),
    "P/L": st.column_config.NumberColumn(format="$%.2f"),
    "return": st.column_config.NumberColumn(format="percent"),
    "weight": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
})

c1, c2 = st.columns(2)
donut = go.Figure(go.Pie(labels=by_ticker.index, values=by_ticker["value"], hole=0.55))
donut.update_layout(hovermode="closest")
with c1:
    ui.chart(donut, 360, "Allocation")
bt = by_ticker.sort_values("return")
bars = go.Figure(go.Bar(x=bt["return"] * 100, y=bt.index, orientation="h",
                        marker_color=[ui.UP if r >= 0 else ui.DOWN for r in bt["return"]],
                        text=[f"{r:+.1%}" for r in bt["return"]], textposition="outside"))
bars.update_layout(xaxis_title="Return (%)", hovermode="closest")
with c2:
    ui.chart(bars, 360, "Return by holding")

st.subheader("Portfolio risk (last 12 months)")
tickers = list(by_ticker.index)
closes = data.get_closes(tickers, period="1Y")
if closes.shape[1] == len(tickers) and len(closes) > 30:
    w = by_ticker.loc[closes.columns, "weight"].values
    rets = risk.returns(closes)
    stats = risk.portfolio_summary(rets, w)
    cols = st.columns(len(stats))
    for col, (k, v) in zip(cols, stats.items()):
        col.metric(k, f"{v:.2f}" if "ratio" in k else ui.pct(v))

    st.subheader("Monte Carlo: portfolio value in 30 trading days")
    port = risk.portfolio_returns(rets, w)
    paths = risk.monte_carlo_paths(total_value, port, days=30, n_paths=2000)
    bands = risk.percentile_bands(paths)
    x = np.arange(len(bands))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=bands["p95"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=bands["p5"], fill="tonexty", fillcolor="rgba(56,189,248,0.12)",
                             line=dict(width=0), name="5–95%"))
    fig.add_trace(go.Scatter(x=x, y=bands["p75"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=bands["p25"], fill="tonexty", fillcolor="rgba(56,189,248,0.25)",
                             line=dict(width=0), name="25–75%"))
    fig.add_trace(go.Scatter(x=x, y=bands["p50"], line=dict(color=ui.ACCENT, width=2), name="Median"))
    fig.update_layout(xaxis_title="Trading days ahead", yaxis_title="Portfolio value ($)")
    ui.chart(fig, 380)
    end_vals = paths[-1]
    st.caption(f"Median outcome {ui.money(np.median(end_vals))}. There is a 5% chance of ending below "
               f"{ui.money(np.percentile(end_vals, 5))} ({np.percentile(end_vals, 5) / total_value - 1:+.1%}). "
               "This assumes the past year's volatility continues; it is not a forecast.")
ui.data_status()
