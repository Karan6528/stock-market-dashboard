import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import data, ui

ui.header("Volume & Liquidity", "Trading volume trends, unusual-volume days and turnover.")

c1, c2 = st.columns([1, 2])
with c1:
    symbol = ui.symbol_input("Ticker", "AAPL", key="vol_symbol")
with c2:
    start, end = ui.date_range(default_years=1, key="vol")
window = st.slider("Moving-average window (days)", 5, 60, 20)

df = data.get_history(symbol, start, end)
if len(df) < window + 5:
    st.error("Not enough data in this range.")
    st.stop()

vol = df["Volume"].astype(float)
ma = vol.rolling(window).mean()
rel = vol / ma
spikes = rel > 2

fig = go.Figure()
fig.add_trace(go.Bar(x=df.index, y=vol, name="Volume", marker_color=np.where(spikes, "#F59E0B", "#475569")))
fig.add_trace(go.Scatter(x=df.index, y=ma, name=f"{window}-day average", line=dict(color=ui.ACCENT, width=2)))
fig.update_layout(yaxis_title="Shares")
ui.chart(fig, 420, f"{symbol} daily volume (orange = more than 2× average)")

dollar_vol = (df["Close"] * vol).tail(window).mean()
info, _ = data.get_info(symbol)
shares_out = info.get("sharesOutstanding")
m = st.columns(4)
m[0].metric("Average volume", ui.big_number(vol.tail(window).mean()))
m[1].metric("Average $ volume", "$" + ui.big_number(dollar_vol))
m[2].metric("Volume vs average (today)", f"{rel.iloc[-1]:.2f}×")
m[3].metric("Daily turnover", ui.pct(vol.tail(window).mean() / shares_out, 3) if shares_out else "n/a",
            help="Average daily volume ÷ shares outstanding.")
if not shares_out:
    st.caption("Turnover needs shares outstanding from company info (available online, or offline once cached).")

st.subheader("Does volume confirm price moves?")
ret = df["Close"].pct_change() * 100
sc = go.Figure(go.Scatter(x=rel, y=ret, mode="markers", marker=dict(color=ret, colorscale="RdYlGn", cmid=0, size=7),
                          text=df.index.strftime("%Y-%m-%d"),
                          hovertemplate="%{text}<br>Rel. volume %{x:.2f}×<br>Return %{y:.2f}%<extra></extra>"))
sc.update_layout(xaxis_title="Relative volume (× average)", yaxis_title="Daily return (%)", hovermode="closest")
ui.chart(sc, 380)
corr = np.corrcoef(rel.dropna(), ret.loc[rel.dropna().index].abs().fillna(0))[0, 1]
st.caption(f"Correlation between relative volume and the size of the daily move: **{corr:.2f}**. Big moves on high "
           "volume are generally considered more meaningful than big moves on thin volume.")

top = df.assign(**{"Rel. volume": rel, "Return %": ret}).loc[spikes].sort_values("Rel. volume", ascending=False)
with st.expander(f"{int(spikes.sum())} unusual-volume days"):
    st.dataframe(top[["Close", "Volume", "Rel. volume", "Return %"]].round(2), width="stretch")
ui.data_status()
