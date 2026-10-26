"""Sentiment scoring with graceful fallbacks.

VADER is tuned for short, informal text like headlines and social posts, so it
is preferred. TextBlob is the fallback, then a tiny finance word list so the
feature never breaks.
"""
from __future__ import annotations

import re

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
except Exception:  # pragma: no cover
    _vader = None

try:
    from textblob import TextBlob
except Exception:  # pragma: no cover
    TextBlob = None

_POS = {"beat", "beats", "surge", "surges", "soar", "soars", "record", "growth", "upgrade", "upgraded", "strong",
        "gain", "gains", "rally", "rallies", "profit", "bullish", "outperform", "raises", "jump", "jumps", "great"}
_NEG = {"miss", "misses", "plunge", "plunges", "drop", "drops", "fall", "falls", "downgrade", "downgraded", "weak",
        "loss", "losses", "lawsuit", "probe", "bearish", "cut", "cuts", "slump", "recall", "fraud", "layoffs"}


def engine() -> str:
    return "VADER" if _vader else "TextBlob" if TextBlob else "keyword list"


def score(text: str) -> float:
    """Polarity in [-1, 1]."""
    text = text or ""
    if _vader:
        return float(_vader.polarity_scores(text)["compound"])
    if TextBlob:
        return float(TextBlob(text).sentiment.polarity)
    words = re.findall(r"[a-z]+", text.lower())
    p, n = sum(w in _POS for w in words), sum(w in _NEG for w in words)
    return 0.0 if p + n == 0 else (p - n) / (p + n)


def label(polarity: float, threshold: float = 0.05) -> str:
    if polarity > threshold:
        return "Positive"
    if polarity < -threshold:
        return "Negative"
    return "Neutral"


EMOJI = {"Positive": "😊", "Neutral": "😐", "Negative": "😟"}
VERDICT = {"Positive": "Good news", "Neutral": "Neutral news", "Negative": "Bad news"}


def emoji(polarity: float) -> str:
    """Emoji for a score, with extra emphasis for strong readings."""
    if polarity >= 0.5:
        return "🤩"
    if polarity <= -0.5:
        return "😱"
    return EMOJI[label(polarity)]


def verdict(polarity: float) -> str:
    return f"{emoji(polarity)} {VERDICT[label(polarity)]}"


def parse_published(value):
    """Parse yfinance ISO timestamps and Alpha Vantage's 20251025T120000 format."""
    import pandas as pd

    s = str(value or "").strip()
    if not s:
        return pd.NaT
    for fmt in (None, "%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
        try:
            ts = pd.to_datetime(s, format=fmt, utc=True)
            return ts.tz_convert("America/New_York").tz_localize(None)
        except (ValueError, TypeError):
            continue
    return pd.NaT


def price_reaction(published, prices):
    """How the stock moved after a headline.

    News published after the 16:00 New York close can only affect the next
    session. Returns (reaction day, 1-day return, 3-day return); the returns
    are NaN when the reaction day is not in the data yet.
    """
    import numpy as np
    import pandas as pd

    if pd.isna(published) or prices is None or len(prices) < 2:
        return pd.NaT, np.nan, np.nan
    day = published.normalize()
    if published.hour >= 16:
        day += pd.Timedelta(days=1)
    pos = prices.index.searchsorted(day)
    if pos >= len(prices) or pos == 0:
        return pd.NaT, np.nan, np.nan
    prev = prices.iloc[pos - 1]
    r1 = prices.iloc[pos] / prev - 1
    r3 = prices.iloc[min(pos + 2, len(prices) - 1)] / prev - 1 if pos + 2 < len(prices) else np.nan
    return prices.index[pos], float(r1), float(r3) if pd.notna(r3) else np.nan
