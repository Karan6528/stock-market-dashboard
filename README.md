<p align="center">
  <img src="assets/banner_wide.jpg" alt="Stock Market Dashboard banner" width="100%">
</p>

# Stock Market Dashboard

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://karan-stock-dashboard.streamlit.app)

**Live app:** https://karan-stock-dashboard.streamlit.app

I built this during my MSc at Gisma because I wanted one place to look at a stock from every angle:
the chart, the news, the risk, and whether any of the "AI predicts the market" stuff actually holds up.
It started as about 40 separate Streamlit scripts that launched each other. This repo is the cleaned-up
version: one app, 22 pages, and it still works when I'm offline on a train.

> The free Streamlit server goes to sleep when nobody uses it. If you see "This app has gone to sleep",
> click the button and give it about 30 seconds to wake up.

## What it does

**Market:** live quote with an intraday chart, index trends (S&P 500, Nasdaq, Dow, Russell), sector ETF
performance, and a screener that filters a watchlist by price, volume, RSI and trend.

**Charts:** candlestick / OHLC charts, a "what if I had invested X in year Y" calculator, multi-stock
comparison with correlation heatmaps, 10 technical indicators (RSI, MACD, Bollinger, ATR, ADX, ...),
volume analysis, and anomaly detection with a rolling z-score or an Isolation Forest.

**News:** pulls recent headlines, scores each one with VADER, and tags it 😊 good, 😐 neutral or 😟 bad.
Then it checks what the stock price actually did in the next session, so you can see whether the market
agreed with the sentiment. (Often it doesn't, which was an interesting finding in itself.)

**Trade Planner:** you enter a budget and how much you're willing to lose. It tells you how many shares
to buy, where to put a stop-loss (based on ATR), a target price, and how many days to hold. The holding
period comes from replaying the same plan on every trading day in the last 10 years and measuring how
long the target took to hit.

**Portfolio:** a holdings tracker saved in SQLite, portfolio analytics with an efficient frontier
(max-Sharpe and min-volatility weights), VaR / CVaR, beta, Monte Carlo simulations, a GARCH(1,1)
volatility forecast, and a backtester for five simple strategies against buy-and-hold, including
trading costs.

**Machine learning:**
- Next-day prediction with 8 models (linear, ridge, trees, random forest, gradient boosting, KNN, SVR,
  XGBoost), all compared against just guessing "tomorrow = today".
- A volatility-regime classifier (low / medium / high for the next month).
- A PPO reinforcement-learning agent trained on one period and tested on a later one.

**Alerts:** price alerts that check while the page is open, with optional email.

## A few things I learned (and fixed)

The first version of this project had some results that looked great but were wrong, so these
are the rules I follow now:

- **Compare against a dumb baseline.** My early models had 99% "accuracy" predicting prices. It turns out
  predicting today's price for tomorrow also gets 99%. Now every model is compared with that naive
  guess, and most of them don't beat it, which is the honest result for daily stock prices.
- **No peeking at the future.** My old risk classifier scored ~100% because the label was hidden in the
  features. Features now only use data up to day *t*, and there's a unit test that changes future prices
  and checks that past features stay the same.
- **Train on the past, test on the future.** No random shuffling of time series.
- **Show a range, not one number.** Long-term projections are Monte Carlo fan charts, not a single
  straight trend line.

## Offline mode

I wanted it to work without internet and without API keys, so data is fetched in this order:

1. **Live** from Yahoo Finance (`yfinance`). Every download gets saved to `data/`.
2. **Cache:** if you're offline (toggle in the sidebar), it reads from `data/`.
3. **Demo:** if a ticker was never downloaded, it generates synthetic prices so the page still works.
   There's always a banner when you're looking at demo data.

To download a bunch of tickers in one go:

```bash
python scripts/download_data.py                 # default watchlist, indices and sector ETFs
python scripts/download_data.py AAPL TSLA SAP.DE --years 5
```

## Running it locally

```bash
git clone https://github.com/Karan6528/stock-market-dashboard.git
cd stock-market-dashboard

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac / Linux

pip install -r requirements.txt
streamlit run app.py
```

It opens at http://localhost:8501.

The RL page needs PyTorch, which is big (about 1 GB), so it's kept separate:
`pip install -r requirements-rl.txt`

Tests: `pip install -r requirements-dev.txt` and then `pytest -q`

## API keys (optional)

None are needed to run the app. If you want the extras, copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml` and fill in:

- `ALPHA_VANTAGE_API_KEY`: news with Alpha Vantage's own sentiment scores (otherwise it uses Yahoo / Google News)
- `YOUTUBE_API_KEY`: the YouTube videos page
- `SMTP_USER` / `SMTP_PASSWORD`: email alerts (use a Gmail app password)

`secrets.toml` is in `.gitignore`, so keys never end up on GitHub.

## Project structure

```
app.py                 navigation + offline toggle
core/
  data.py              live -> cache -> demo data, news from 4 sources
  indicators.py        RSI, MACD, Bollinger, ATR, ADX, stochastic, CCI, OBV
  risk.py              Sharpe, VaR/CVaR, beta, drawdown, Monte Carlo, optimiser, GARCH
  ml.py                features, models, evaluation against the naive baseline
  planner.py           position sizing + historical replay for the trade planner
  sentiment.py         VADER scoring, emoji verdicts, price reaction to news
  rl_env.py            Gymnasium environment for the PPO agent
  portfolio_db.py      SQLite holdings
  ui.py                shared widgets and chart styling
views/                 one file per page
scripts/               data download script
tests/                 unit tests
```

## Tech

Python, Streamlit, pandas, NumPy, SciPy, scikit-learn, XGBoost, arch, Plotly, yfinance, VADER,
Stable-Baselines3, SQLite

## Disclaimer

This is a learning project, not financial advice. Please don't trade real money based on a student's
dashboard 🙂

---

Karan Lodha · MSc Data Science, AI & Digital Business, Gisma University of Applied Sciences, Berlin
[GitHub](https://github.com/Karan6528) · [LinkedIn](https://www.linkedin.com/in/karanlodha/)
