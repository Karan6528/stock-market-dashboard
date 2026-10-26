"""AI Stock Market Dashboard: entry point.

Run locally:   streamlit run app.py
"""
import streamlit as st

from core import config, data

st.set_page_config(
    page_title="AI Stock Market Dashboard",
    page_icon=":material/monitoring:",
    layout="wide",
)

P = "views"
pages = {
    "Overview": [
        st.Page(f"{P}/home.py", title="Home", icon=":material/home:", default=True),
    ],
    "Market": [
        st.Page(f"{P}/live_quote.py", title="Live Quote", icon=":material/bolt:"),
        st.Page(f"{P}/market_trends.py", title="Market & Sectors", icon=":material/public:"),
        st.Page(f"{P}/screener.py", title="Stock Screener", icon=":material/filter_alt:"),
    ],
    "Charts": [
        st.Page(f"{P}/price_charts.py", title="Price Charts", icon=":material/candlestick_chart:"),
        st.Page(f"{P}/compare.py", title="Compare & Correlate", icon=":material/compare_arrows:"),
        st.Page(f"{P}/technical_indicators.py", title="Technical Indicators", icon=":material/insights:"),
        st.Page(f"{P}/volume.py", title="Volume & Liquidity", icon=":material/bar_chart:"),
        st.Page(f"{P}/anomalies.py", title="Anomaly Detection", icon=":material/crisis_alert:"),
    ],
    "Fundamentals & News": [
        st.Page(f"{P}/financials.py", title="Financials", icon=":material/account_balance:"),
        st.Page(f"{P}/news_sentiment.py", title="News Sentiment 😊😟", icon=":material/newspaper:"),
        st.Page(f"{P}/videos.py", title="YouTube Videos", icon=":material/smart_display:"),
    ],
    "Portfolio": [
        st.Page(f"{P}/trade_planner.py", title="Trade Planner", icon=":material/calculate:"),
        st.Page(f"{P}/portfolio_tracker.py", title="My Portfolio", icon=":material/account_balance_wallet:"),
        st.Page(f"{P}/portfolio_analytics.py", title="Portfolio Analytics", icon=":material/pie_chart:"),
        st.Page(f"{P}/risk_dashboard.py", title="Risk Dashboard", icon=":material/shield:"),
        st.Page(f"{P}/backtest.py", title="Strategy Backtest", icon=":material/history:"),
    ],
    "Machine Learning": [
        st.Page(f"{P}/ml_prediction.py", title="Next-Day Prediction", icon=":material/model_training:"),
        st.Page(f"{P}/long_term_projection.py", title="Long-Term Projection", icon=":material/trending_up:"),
        st.Page(f"{P}/risk_classifier.py", title="Volatility Regimes", icon=":material/psychology:"),
        st.Page(f"{P}/rl_trading.py", title="RL Trading Agent", icon=":material/smart_toy:"),
    ],
    "Alerts": [
        st.Page(f"{P}/price_alerts.py", title="Price Alerts", icon=":material/notifications_active:"),
    ],
}

with st.sidebar:
    st.markdown("### :material/monitoring: AI Stock Dashboard")
    st.toggle(
        "Offline mode",
        value=config.default_offline(),
        key="offline_mode",
        help="Use only cached prices in data/ (or demo data). No internet or API keys needed.",
    )
    if data.yf is None:
        st.caption(":material/warning: `yfinance` is not installed, so only cached or demo data is available.")
    elif st.session_state.get("offline_mode"):
        n = len(data.cached_symbols())
        st.caption(f":material/cloud_off: Offline. {n} symbol(s) cached in `data/`.")
    else:
        st.caption(":material/cloud_done: Online via Yahoo Finance. Downloads are cached for offline use.")

nav = st.navigation(pages, position="sidebar", expanded=True)

with st.sidebar:
    st.divider()
    st.caption("Built by **Karan Lodha** · [GitHub](https://github.com/Karan6528) · "
               "[LinkedIn](https://www.linkedin.com/in/karanlodha/)")
    st.caption("Educational project, not investment advice.")

nav.run()
