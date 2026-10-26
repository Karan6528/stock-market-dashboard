import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import data, ui

ui.header("Financials", "Company profile, valuation, profitability and the income statement.")

symbol = ui.symbol_input("Ticker", "AAPL", key="fin_symbol")
info, source = data.get_info(symbol)

if not info:
    st.info(
        f"No company data for {symbol} yet. Fundamentals come from Yahoo Finance: open this page once while "
        "online and they're cached for offline use.",
        icon=":material/cloud_off:",
    )
    st.stop()
if source == "cache":
    st.caption(":material/cloud_off: Showing cached company data.")

name = info.get("longName") or info.get("shortName") or symbol
st.subheader(name)
meta = [info.get("sector"), info.get("industry"), info.get("country"), info.get("exchange")]
st.caption(" · ".join(str(x) for x in meta if x))


def val(key, fmt="num"):
    v = info.get(key)
    if v is None or v == "N/A":
        return "n/a"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if fmt == "pct":
        return ui.pct(v)
    if fmt == "money":
        return ui.money(v)
    if fmt == "big":
        return "$" + ui.big_number(v)
    if fmt == "x":
        return f"{v:.2f}×"
    return f"{v:,.2f}"


groups = {
    "Valuation": [("Market cap", "marketCap", "big"), ("Trailing P/E", "trailingPE", "x"),
                  ("Forward P/E", "forwardPE", "x"), ("Price / book", "priceToBook", "x"),
                  ("EV / EBITDA", "enterpriseToEbitda", "x"), ("PEG ratio", "trailingPegRatio", "x")],
    "Profitability": [("Revenue (TTM)", "totalRevenue", "big"), ("EBITDA", "ebitda", "big"),
                      ("Net income", "netIncomeToCommon", "big"), ("Gross margin", "grossMargins", "pct"),
                      ("Operating margin", "operatingMargins", "pct"), ("Profit margin", "profitMargins", "pct")],
    "Growth & returns": [("Revenue growth (YoY)", "revenueGrowth", "pct"),
                         ("Earnings growth (QoQ YoY)", "earningsQuarterlyGrowth", "pct"),
                         ("Return on equity", "returnOnEquity", "pct"), ("Return on assets", "returnOnAssets", "pct"),
                         ("Trailing EPS", "trailingEps", "money"), ("Forward EPS", "forwardEps", "money")],
    "Balance sheet & dividends": [("Total cash", "totalCash", "big"), ("Total debt", "totalDebt", "big"),
                                  ("Debt / equity", "debtToEquity", "num"), ("Current ratio", "currentRatio", "num"),
                                  ("Dividend yield", "dividendYield", "num"), ("Beta", "beta", "num")],
}
for title, items in groups.items():
    st.markdown(f"**{title}**")
    cols = st.columns(len(items))
    for col, (label, key, fmt) in zip(cols, items):
        col.metric(label, val(key, fmt))
st.caption("Dividend yield is shown as Yahoo reports it (a percentage in recent yfinance versions).")

target = info.get("targetMeanPrice")
price = info.get("currentPrice") or info.get("regularMarketPrice")
if target and price:
    st.markdown("**Analyst consensus**")
    c = st.columns(4)
    c[0].metric("Current price", ui.money(price))
    c[1].metric("Mean target", ui.money(target), ui.pct(target / price - 1, 1, sign=True))
    c[2].metric("Target range", f"{ui.money(info.get('targetLowPrice'), 0)} – {ui.money(info.get('targetHighPrice'), 0)}")
    c[3].metric("Recommendation", str(info.get("recommendationKey", "n/a")).replace("_", " ").title(),
                f"{info.get('numberOfAnalystOpinions', '?')} analysts", delta_color="off")

stmt, s_src = data.get_income_statement(symbol)
if not stmt.empty:
    st.subheader("Income statement (annual)")
    wanted = ["Total Revenue", "Gross Profit", "Operating Income", "EBITDA", "Net Income"]
    rows = [r for r in wanted if r in stmt.index]
    tbl = stmt.loc[rows].apply(pd.to_numeric, errors="coerce")
    tbl.columns = [str(c)[:4] for c in tbl.columns]
    tbl = tbl[sorted(tbl.columns)]
    fig = go.Figure()
    for r in rows:
        fig.add_trace(go.Bar(x=tbl.columns, y=tbl.loc[r] / 1e9, name=r))
    fig.update_layout(barmode="group", yaxis_title="USD billions", hovermode="x")
    ui.chart(fig, 400)
    st.dataframe((tbl / 1e9).round(2).rename(index=lambda x: f"{x} ($B)"), width="stretch")

if info.get("longBusinessSummary"):
    with st.expander("Business summary"):
        st.write(info["longBusinessSummary"])
