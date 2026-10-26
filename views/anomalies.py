import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest

from core import data, ui

ui.header("Anomaly Detection", "Find unusual trading days with a rolling z-score or an Isolation Forest model.")

c1, c2 = st.columns([1, 2])
with c1:
    symbol = ui.symbol_input("Ticker", "AAPL", key="an_symbol")
with c2:
    start, end = ui.date_range(default_years=2, key="an")

method = st.radio("Method", ["Rolling z-score", "Isolation Forest"], horizontal=True)
df = data.get_history(symbol, start, end)
if len(df) < 80:
    st.error("Need at least ~80 trading days.")
    st.stop()

ret = df["Close"].pct_change()
if method == "Rolling z-score":
    c1, c2 = st.columns(2)
    window = c1.slider("Lookback window (days)", 20, 250, 60,
                       help="Each day is compared with the previous N days only, so the score uses no future information.")
    threshold = c2.slider("Threshold (|z|)", 2.0, 5.0, 3.0, 0.5)
    mu = ret.rolling(window).mean().shift(1)
    sd = ret.rolling(window).std().shift(1)
    score = (ret - mu) / sd
    flags = score.abs() > threshold
    score_name = "z-score"
else:
    c1, c2 = st.columns(2)
    contamination = c1.slider("Expected share of anomalies", 0.005, 0.1, 0.02, 0.005)
    vol_z = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()
    feats = pd.DataFrame({
        "return": ret,
        "range": (df["High"] - df["Low"]) / df["Close"],
        "gap": df["Open"] / df["Close"].shift() - 1,
        "volume_z": vol_z,
    }).dropna()
    c2.caption("Features: daily return, intraday range, overnight gap and volume z-score.")
    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=300)
    labels = model.fit_predict(feats)
    score = pd.Series(-model.score_samples(feats), index=feats.index).reindex(df.index)
    flags = pd.Series(labels == -1, index=feats.index).reindex(df.index, fill_value=False)
    score_name = "anomaly score"

flags = flags.fillna(False).astype(bool)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color=ui.ACCENT, width=1.6)))
fig.add_trace(go.Scatter(
    x=df.index[flags], y=df["Close"][flags], mode="markers", name="Anomaly",
    marker=dict(size=11, color=np.where(ret[flags] >= 0, ui.UP, ui.DOWN), line=dict(width=1, color="white")),
    customdata=np.c_[ret[flags] * 100, score[flags]],
    hovertemplate="%{x|%d %b %Y}<br>Return %{customdata[0]:.2f}%<br>Score %{customdata[1]:.2f}<extra></extra>",
))
ui.chart(fig, 460, f"{symbol}: {int(flags.sum())} anomalies detected")

n = int(flags.sum())
share = n / len(df)
m = st.columns(3)
m[0].metric("Anomalies", n)
m[1].metric("Share of days", ui.pct(share, 1))
m[2].metric("Largest move", ui.pct(ret[flags].abs().max() if n else 0, 1))

if n == 0:
    st.success("No unusual days at this sensitivity. Price action has been within its normal range.")
elif share < 0.01:
    st.info("A handful of outliers. These are typically earnings releases, guidance changes or macro shocks.")
else:
    st.warning("Frequent outliers: this stock has had a volatile period. Consider wider stops or smaller position sizes.")

if n:
    table = pd.DataFrame({"Close": df["Close"], "Return %": ret * 100, score_name: score,
                          "Volume": df["Volume"]})[flags].sort_index(ascending=False)
    st.dataframe(table.round(2), width="stretch")
ui.data_status()
