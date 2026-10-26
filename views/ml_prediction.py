import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import data, ml, ui

ui.header("Next-Day Prediction",
          "Several ML models predict tomorrow's return from today's indicators, and every model is benchmarked "
          "against a naive 'no change' forecast.")

zoo = list(ml.model_zoo())
c1, c2 = st.columns([1, 2])
with c1:
    symbol = ui.symbol_input("Ticker", "AAPL", key="ml_symbol")
    period = st.selectbox("Training history", ["2Y", "5Y", "10Y"], index=1)
    test_size = st.slider("Test set (most recent %)", 10, 40, 20) / 100
with c2:
    chosen = st.multiselect("Models", zoo, default=[m for m in zoo if m in
                                                    ("Ridge Regression", "Random Forest", "Gradient Boosting", "XGBoost")])
if not chosen:
    st.info("Pick at least one model.")
    st.stop()

df = data.get_history(symbol, period=period)
if len(df) < 300:
    st.error("Need at least ~300 trading days. Choose a longer history.")
    st.stop()


@st.cache_data(show_spinner=False)
def run(df_: pd.DataFrame, models: tuple, test: float):
    return ml.evaluate(df_, list(models), test)


with st.spinner("Training models…"):
    res = run(df, tuple(chosen), test_size)

metrics = res["metrics"]
best = metrics.drop("Naive (no change)")["MAE ($)"].idxmin()
beat = metrics.loc[best, "vs naive"] < 0

st.subheader("Next trading day forecast")
cols = st.columns(min(len(res["forecasts"]), 4))
for i, (name, f) in enumerate(res["forecasts"].items()):
    cols[i % len(cols)].metric(name, ui.money(f["price"]), f"{f['return']:+.2%}")
st.caption(f"Based on data up to {res['last_date']:%d %b %Y} (last close {ui.money(res['last_close'])}). "
           f"Trained on {res['n_train']} days, tested on the most recent {res['n_test']} days.")

st.subheader("How good is it really?")
st.dataframe(metrics, width="stretch", column_config={
    "MAE ($)": st.column_config.NumberColumn(format="$%.3f"),
    "RMSE ($)": st.column_config.NumberColumn(format="$%.3f"),
    "Return MAE (%)": st.column_config.NumberColumn(format="%.3f"),
    "Direction accuracy": st.column_config.NumberColumn(format="percent"),
    "vs naive": st.column_config.NumberColumn("Error vs naive", format="percent",
                                              help="Negative = smaller error than predicting no change."),
})
if beat:
    st.success(f"**{best}** had {abs(metrics.loc[best, 'vs naive']):.1%} lower error than the naive forecast on "
               "the test set. Check whether this holds for other tickers and periods before reading much into it.")
else:
    st.warning("No model beat the naive 'tomorrow = today' forecast on the test set. This is the normal "
               "result for daily stock returns, which are very close to unpredictable.")
st.caption(f"Up days in the test set: {res['up_share']:.0%}. A direction accuracy near that number means the model "
           "is mostly guessing 'up'.")

st.subheader("Actual vs predicted (test set)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=res["actual"].index, y=res["actual"], name="Actual", line=dict(color="white", width=2)))
for name, s in res["predictions"].items():
    if name == "Naive (no change)":
        continue
    fig.add_trace(go.Scatter(x=s.index, y=s, name=name, line=dict(width=1), opacity=0.8))
ui.chart(fig, 460)
st.caption("Predicted prices track the actual line closely mainly because each prediction starts from "
           "yesterday's real price. That's why the table compares models with the naive baseline.")

with st.expander("Feature importance (Random Forest)"):
    imp = ml.feature_importance(df)
    bar = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=ui.ACCENT))
    bar.update_layout(hovermode="closest")
    ui.chart(bar, 420)
with st.expander("Method"):
    st.markdown("""
- **Target**: next-day return (t → t+1), not the raw price. Predicting raw prices lets any model look
  accurate just by copying today's price.
- **Features** (known at today's close): lagged returns, 5- and 20-day momentum, rolling volatility,
  distance from SMA 10/50, RSI, MACD histogram, intraday range and a volume z-score.
- **Split**: chronological, with the most recent part held out. No shuffling, so the test set is
  always later than the training data.
- **Baseline**: "no change", i.e. tomorrow's price equals today's. A model that can't beat this has no predictive value.
""")
ui.data_status()
ui.disclaimer()
