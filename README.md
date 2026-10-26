<p align="center">
  <img src="assets/banner_wide.jpg" alt="AI Stock Market Dashboard" width="100%">
</p>

# AI Stock Market Dashboard

A multi-page **Streamlit** app for stock market analysis: live quotes, technical indicators, portfolio risk,
strategy backtesting and machine learning. It runs **offline first**: everything works from a local cache
(or clearly labelled demo data) with no internet connection and no API keys.

**Stack:** Python · Streamlit · pandas · NumPy · SciPy · scikit-learn · XGBoost · arch (GARCH) · Plotly ·
yfinance · Stable-Baselines3 (PPO) · SQLite

---

## Features

| Section | Pages | Highlights |
|---|---|---|
| **Market** | Live Quote · Market & Sectors · Stock Screener | Auto-refreshing intraday chart, index trend checks (EMA/SMA, golden cross), 11 sector ETFs, screener on RSI / trend / Bollinger |
| **Charts** | Price Charts · Compare & Correlate · Technical Indicators · Volume · Anomaly Detection | Candlestick/OHLC, "what if I had invested", rolling correlation, 10 indicators computed locally, Isolation Forest + rolling z-score anomalies |
| **Fundamentals & News** | Financials · News Sentiment · YouTube Videos | Valuation and profitability ratios, income statement, analyst targets, NLP headline sentiment with 😊 / 😐 / 😟 verdicts checked against the next-session price reaction, transcript summaries |
| **Portfolio** | Trade Planner · My Portfolio · Portfolio Analytics · Risk Dashboard · Strategy Backtest | Risk-based position sizing (shares to buy, ATR stop-loss, target, holding days from historical replay), SQLite-backed holdings, efficient frontier (max-Sharpe / min-vol), VaR & CVaR, beta/alpha, Monte Carlo fan charts, GARCH(1,1), 5 strategies vs buy-and-hold with costs |
| **Machine Learning** | Next-Day Prediction · Long-Term Projection · Volatility Regimes · RL Trading Agent | 8 models benchmarked against a naive baseline, leak-free features, regime classifier, PPO agent evaluated out-of-sample |
| **Alerts** | Price Alerts | Threshold and %-move alerts with optional email |

## Quick start (local, offline-first)

```bash
git clone https://github.com/Karan6528/stockmarketprediction.git
cd stockmarketprediction

python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

The app opens at http://localhost:8501.

**Offline mode.** Switch on *Offline mode* in the sidebar, or set `OFFLINE_MODE = true` in `.streamlit/secrets.toml`.

1. **Online**: prices come from Yahoo Finance via `yfinance`. Every download is saved to `data/prices/*.csv`
   (company info, financials and news are cached too).
2. **Offline**: pages read only from `data/`.
3. **No cache yet**: the app falls back to **demo data**. These are synthetic, deterministic prices with a shared
   market factor, so correlations and betas behave realistically. A banner always says when you're
   looking at cached or demo data.

To fill the cache in one go while online:

```bash
python scripts/download_data.py                     # default watchlist + indices + sector ETFs, 10 years
python scripts/download_data.py AAPL TSLA SAP.DE --years 5
```

**Optional extras**

```bash
pip install -r requirements-rl.txt     # RL Trading Agent page (PyTorch, ~1 GB)
pip install -r requirements-dev.txt    # pytest
pytest -q                              # 15 unit tests
```

## API keys (all optional)

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in what you need.
`secrets.toml` is git-ignored, and **no key is hardcoded anywhere in the code**.

| Key | Used for | Without it |
|---|---|---|
| `ALPHA_VANTAGE_API_KEY` | News with provider sentiment scores | Falls back to Yahoo Finance headlines |
| `YOUTUBE_API_KEY` | YouTube Videos page | Page shows setup instructions |
| `SMTP_USER`, `SMTP_PASSWORD` | Email price alerts | In-app alerts only |

## Deploy online (Streamlit Community Cloud)

1. Push the repo to GitHub (secrets and `data/` are excluded by `.gitignore`).
2. On [share.streamlit.io](https://share.streamlit.io), click **Create app**, pick the repo, and set the main file to `app.py`.
3. Under **Advanced settings → Secrets**, paste the contents of your `secrets.toml`.
4. Deploy. The cloud disk is temporary, so the cache rebuilds as people use the app.

The RL page needs `requirements-rl.txt`. Cloud builds use `requirements.txt` only, so the RL page shows
install instructions there, which keeps the deployment small.

## Project structure

```
app.py                  # entry point: navigation + offline toggle
core/
  config.py             # paths, secrets, offline switch
  data.py               # live → cache → demo fallback chain
  indicators.py         # SMA, EMA, RSI (Wilder), MACD, Bollinger, ATR, ADX, Stochastic, CCI, OBV
  risk.py               # returns, Sharpe/Sortino, VaR/CVaR, beta, drawdown, Monte Carlo, optimiser, GARCH
  ml.py                 # leak-free features, model zoo, evaluation vs naive baseline
  rl_env.py             # Gymnasium trading environment
  sentiment.py          # VADER → TextBlob → keyword fallback, emoji verdicts, price reaction
  planner.py            # position sizing, stop/target, historical plan replay
  portfolio_db.py       # SQLite holdings
  ui.py                 # shared widgets, chart styling, data-source banner
views/                  # one file per page (22 pages)
scripts/download_data.py
tests/test_core.py
```

## Methodology notes

These are the choices that make the results trustworthy rather than just impressive-looking:

- **No look-ahead.** ML features on day *t* use only data up to *t*, and the target is the *t → t+1* return.
  A unit test shocks future prices and asserts that past features don't change. Backtests trade on the
  previous day's signal.
- **Honest baselines.** Next-day models are scored against "tomorrow = today". The volatility classifier is
  scored against a persistence baseline, with regime thresholds taken from training data only.
- **Chronological splits.** Train on the past, test on the future, with no shuffling.
- **Ranges instead of point forecasts.** Long-term projections are Monte Carlo fan charts. A straight-line
  trend extrapolation is shown alongside for comparison, to explain why it misleads.
- **Costs included.** Backtests and the RL environment charge a per-trade cost.

## Disclaimer

This is an educational project and not investment advice.

---

Built by **Karan Lodha**, MSc Data Science, AI & Digital Business @ Gisma University of Applied Sciences, Berlin ·
[GitHub](https://github.com/Karan6528) · [LinkedIn](https://www.linkedin.com/in/karanlodha/)
