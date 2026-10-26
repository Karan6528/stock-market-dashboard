import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import config, data, indicators as ind, ui

ui.header("Stock Screener", "Filter a watchlist by price, volume, RSI and trend.")

universe = ui.symbols_input("Watchlist", ", ".join(config.DEFAULT_SYMBOLS + ["AMD", "NFLX", "JPM", "KO"]),
                            key="scr_universe")

with st.expander("Filters", expanded=True, icon=":material/tune:"):
    c1, c2, c3 = st.columns(3)
    price_min, price_max = c1.slider("Price range ($)", 0, 2000, (0, 2000), step=10)
    min_volume = c2.number_input("Min avg. volume (20d)", 0, 500_000_000, 1_000_000, step=500_000)
    rsi_lo, rsi_hi = c3.slider("RSI (14) between", 0, 100, (0, 100))
    c4, c5 = st.columns(2)
    trend = c4.selectbox("Trend filter", ["Any", "Price above SMA 50", "Price below SMA 50",
                                          "SMA 50 above SMA 200"])
    bb = c5.selectbox("Bollinger filter", ["Any", "Below lower band (oversold)", "Above upper band (overbought)"])


def snapshot(symbols) -> pd.DataFrame:
    rows = []
    for s in symbols:
        df = data.get_history(s, period="1Y")
        if len(df) < 60:
            continue
        c = df["Close"]
        bands = ind.bollinger(c)
        sma200 = ind.sma(c, 200).iloc[-1]
        rows.append({
            "Ticker": s,
            "Price": c.iloc[-1],
            "1D %": c.pct_change().iloc[-1] * 100,
            "1M %": (c.iloc[-1] / c.iloc[-22] - 1) * 100,
            "Avg volume": df["Volume"].tail(20).mean(),
            "RSI": ind.rsi(c).iloc[-1],
            "SMA 50": ind.sma(c, 50).iloc[-1],
            "SMA 200": sma200,
            "BB upper": bands["Upper"].iloc[-1],
            "BB lower": bands["Lower"].iloc[-1],
        })
    return pd.DataFrame(rows)


with st.spinner("Scanning watchlist…"):
    snap = snapshot(universe)

if snap.empty:
    st.warning("No data for these tickers.")
    st.stop()

mask = (snap["Price"].between(price_min, price_max) & (snap["Avg volume"] >= min_volume)
        & snap["RSI"].between(rsi_lo, rsi_hi))
if trend == "Price above SMA 50":
    mask &= snap["Price"] > snap["SMA 50"]
elif trend == "Price below SMA 50":
    mask &= snap["Price"] < snap["SMA 50"]
elif trend == "SMA 50 above SMA 200":
    mask &= snap["SMA 50"] > snap["SMA 200"]
if bb.startswith("Below"):
    mask &= snap["Price"] < snap["BB lower"]
elif bb.startswith("Above"):
    mask &= snap["Price"] > snap["BB upper"]

hits = snap[mask].sort_values("1M %", ascending=False)
st.subheader(f"{len(hits)} of {len(snap)} stocks match")
st.dataframe(
    hits, hide_index=True, width="stretch",
    column_config={
        "Price": st.column_config.NumberColumn(format="$%.2f"),
        "1D %": st.column_config.NumberColumn(format="%.2f%%"),
        "1M %": st.column_config.NumberColumn(format="%.2f%%"),
        "Avg volume": st.column_config.NumberColumn(format="compact"),
        "RSI": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
        "SMA 50": st.column_config.NumberColumn(format="$%.2f"),
        "SMA 200": st.column_config.NumberColumn(format="$%.2f"),
        "BB upper": None, "BB lower": None,
    },
)

if not hits.empty:
    fig = go.Figure(go.Scatter(
        x=hits["RSI"], y=hits["1M %"], mode="markers+text", text=hits["Ticker"], textposition="top center",
        marker=dict(size=14, color=hits["1M %"], colorscale="RdYlGn", cmid=0),
        hovertemplate="%{text}<br>RSI %{x:.0f}<br>1M %{y:.1f}%<extra></extra>",
    ))
    fig.add_vrect(x0=0, x1=30, fillcolor=ui.UP, opacity=0.07, line_width=0, annotation_text="oversold")
    fig.add_vrect(x0=70, x1=100, fillcolor=ui.DOWN, opacity=0.07, line_width=0, annotation_text="overbought")
    fig.update_layout(xaxis_title="RSI (14)", yaxis_title="1-month return (%)", hovermode="closest",
                      xaxis_range=[0, 100])
    ui.chart(fig, 420, "Momentum vs RSI")
    ui.to_csv_download(hits, "screener_results.csv")

ui.data_status()
