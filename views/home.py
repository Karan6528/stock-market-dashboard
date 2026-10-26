import plotly.graph_objects as go
import streamlit as st

from core import config, data, ui

st.image(str(config.ASSETS_DIR / "banner_wide.jpg"), width="stretch")
ui.header(
    "AI Stock Market Dashboard",
    "Market data, technical analysis, portfolio risk and machine learning in one Streamlit app. "
    "It works online through Yahoo Finance, or fully offline from cached data.",
)

st.subheader("Market snapshot")
cols = st.columns(len(config.MARKET_INDICES))
for col, (name, sym) in zip(cols, config.MARKET_INDICES.items()):
    df = data.get_history(sym, period="3M")
    if len(df) < 2:
        col.metric(name, "n/a")
        continue
    last, prev = df["Close"].iloc[-1], df["Close"].iloc[-2]
    col.metric(name, f"{last:,.2f}", f"{(last / prev - 1):+.2%}")
    spark = go.Figure(go.Scatter(x=df.index, y=df["Close"], mode="lines",
                                 line=dict(color=ui.UP if last >= df["Close"].iloc[0] else ui.DOWN, width=2)))
    spark = ui.style(spark, height=90)
    spark.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False, hovermode=False)
    spark.update_xaxes(visible=False)
    spark.update_yaxes(visible=False)
    col.plotly_chart(spark, width="stretch", config={"displayModeBar": False})
ui.data_status()

st.subheader("What's inside")
sections = [
    (":material/bolt: Market", "Live quotes, index trends, sector rotation and a technical stock screener."),
    (":material/candlestick_chart: Charts", "Candlesticks, multi-stock comparison, correlations, 10 technical "
                                           "indicators, volume and anomaly detection."),
    (":material/newspaper: Fundamentals & News", "Key financial ratios, income statements, 😊/😟 news "
                                                 "sentiment checked against the price reaction and YouTube video summaries."),
    (":material/account_balance_wallet: Portfolio", "A trade planner (shares to buy, stop, target, holding days), a persistent holdings tracker, efficient frontier, VaR / "
                                                    "CVaR, beta, Monte Carlo and strategy backtests."),
    (":material/model_training: Machine Learning", "Next-day models benchmarked against a naive baseline, "
                                                   "volatility-regime classification, and a PPO reinforcement-learning agent."),
    (":material/notifications_active: Alerts", "Threshold alerts that poll prices while the page is open, with optional email."),
]
c1, c2 = st.columns(2)
for i, (title, text) in enumerate(sections):
    with (c1 if i % 2 == 0 else c2).container(border=True):
        st.markdown(f"**{title}**")
        st.caption(text)

with st.expander("How offline mode works"):
    st.markdown(
        """
- **Online**: prices come from Yahoo Finance (`yfinance`), and every download is saved to `data/prices/*.csv`.
- **Offline**: switch on *Offline mode* in the sidebar. Pages read from `data/`. Symbols that were never
  downloaded fall back to **demo data** (synthetic prices, always labelled), so every page still works.
- Pre-fill the cache in one go: `python scripts/download_data.py AAPL MSFT NVDA`.
- API keys (optional, for news sentiment, YouTube and email alerts) go in `.streamlit/secrets.toml`, never in code.
"""
    )
    cached = data.cached_symbols()
    st.caption(f"Cached symbols ({len(cached)}): {', '.join(cached) if cached else 'none yet'}")

ui.disclaimer()
