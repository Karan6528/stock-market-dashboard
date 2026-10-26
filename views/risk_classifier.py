import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from core import data, ui

ui.header("Volatility Regimes",
          "Classify whether the next month will be a low-, medium- or high-volatility period, using only past data.")

c1, c2, c3 = st.columns(3)
with c1:
    symbol = ui.symbol_input("Ticker", "AAPL", key="rc_symbol")
period = c2.selectbox("History", ["5Y", "10Y"], index=1)
horizon = c3.selectbox("Forecast horizon (trading days)", [10, 20, 60], index=1)

df = data.get_history(symbol, period=period)
if len(df) < 500:
    st.error("Need at least ~2 years of data.")
    st.stop()

ret = df["Close"].pct_change()
feats = pd.DataFrame({
    "vol_5d": ret.rolling(5).std(),
    "vol_20d": ret.rolling(20).std(),
    "vol_60d": ret.rolling(60).std(),
    "vol_ratio": ret.rolling(5).std() / ret.rolling(60).std(),
    "ret_20d": df["Close"].pct_change(20),
    "drawdown_60d": df["Close"] / df["Close"].rolling(60).max() - 1,
    "range_10d": ((df["High"] - df["Low"]) / df["Close"]).rolling(10).mean(),
    "abs_ret_1d": ret.abs(),
})
# Target: realised volatility over the NEXT `horizon` days (strictly future).
future_vol = ret[::-1].rolling(horizon).std()[::-1].shift(-1)
data_ = pd.concat([feats, future_vol.rename("future_vol")], axis=1).dropna()

split = int(len(data_) * 0.75)
train, test = data_.iloc[:split], data_.iloc[split:]
# Regime thresholds come from the TRAINING period only, so no test information leaks in.
q1, q2 = train["future_vol"].quantile([1 / 3, 2 / 3])


def to_label(v):
    return np.select([v <= q1, v <= q2], [0, 1], 2)


y_tr, y_te = to_label(train["future_vol"]), to_label(test["future_vol"])
X_cols = list(feats.columns)
model = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42,
                               class_weight="balanced", n_jobs=-1)
model.fit(train[X_cols], y_tr)
pred = model.predict(test[X_cols])

# Baseline: "next month looks like the last month", i.e. classify current 20d vol with the same thresholds.
baseline = to_label(test["vol_20d"])

names = ["Low", "Medium", "High"]
acc_model = balanced_accuracy_score(y_te, pred)
acc_base = balanced_accuracy_score(y_te, baseline)
m = st.columns(3)
m[0].metric("Model balanced accuracy", ui.pct(acc_model, 1))
m[1].metric("Persistence baseline", ui.pct(acc_base, 1), help="Predict that the next period's regime equals the current one.")
m[2].metric("Random guessing", "33.3%")

latest = feats.dropna().iloc[[-1]]
proba = model.predict_proba(latest[X_cols])[0]
current = names[int(np.argmax(proba))]
st.subheader(f"Forecast for the next {horizon} trading days: **{current} volatility**")
pb = go.Figure(go.Bar(x=names, y=proba * 100, marker_color=[ui.UP, "#F59E0B", ui.DOWN],
                      text=[f"{p:.0%}" for p in proba], textposition="outside"))
pb.update_layout(yaxis_title="Probability (%)", yaxis_range=[0, 105], hovermode="closest")
ui.chart(pb, 300)
st.caption(f"Regime cut-offs (daily vol, from training data): low ≤ {q1:.2%} < medium ≤ {q2:.2%} < high. "
           f"Annualised ≈ {q1 * np.sqrt(252):.0%} and {q2 * np.sqrt(252):.0%}.")

c1, c2 = st.columns(2)
cm = confusion_matrix(y_te, pred, labels=[0, 1, 2])
heat = go.Figure(go.Heatmap(z=cm, x=[f"pred {n}" for n in names], y=[f"true {n}" for n in names],
                            text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
heat.update_layout(hovermode="closest", yaxis_autorange="reversed")
with c1:
    ui.chart(heat, 340, "Confusion matrix (test period)")
imp = pd.Series(model.feature_importances_, index=X_cols).sort_values()
bar = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=ui.ACCENT))
bar.update_layout(hovermode="closest")
with c2:
    ui.chart(bar, 340, "Feature importance")

st.subheader("Regimes over the test period")
tl = test.assign(pred=pred)
fig = go.Figure()
fig.add_trace(go.Scatter(x=tl.index, y=tl["future_vol"] * np.sqrt(252) * 100, name=f"Actual next-{horizon}d vol",
                         line=dict(color="white")))
colors = {0: ui.UP, 1: "#F59E0B", 2: ui.DOWN}
for k in (0, 1, 2):
    sel = tl[tl["pred"] == k]
    fig.add_trace(go.Scatter(x=sel.index, y=sel["future_vol"] * np.sqrt(252) * 100, mode="markers",
                             marker=dict(size=4, color=colors[k]), name=f"Predicted {names[k]}"))
fig.add_hline(y=q1 * np.sqrt(252) * 100, line_dash="dot")
fig.add_hline(y=q2 * np.sqrt(252) * 100, line_dash="dot")
fig.update_layout(yaxis_title="Annualised volatility (%)")
ui.chart(fig, 400)
st.caption("Volatility clusters (calm periods follow calm periods), which makes it far more predictable than "
           "price direction. The original version scored about 100% only because its features contained the "
           "label. Here the target lies strictly in the future and thresholds come only from training data.")
ui.data_status()
