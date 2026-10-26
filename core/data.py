"""Data access with an offline-first fallback chain.

Every request goes through the same three steps:

1. **live**: Yahoo Finance via ``yfinance`` (skipped in offline mode). Successful
   downloads are written to ``data/`` so they are available offline later.
2. **cache**: whatever was saved in ``data/`` earlier.
3. **demo**: synthetic prices, so every page still works with no internet
   and no cache. Pages label demo data clearly.

Which source served each symbol is recorded so pages can tell the user.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import zlib
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import streamlit as st

from . import config

try:  # optional so the app still opens in offline mode without it
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

log = logging.getLogger(__name__)

PRICE_DIR = config.DATA_DIR / "prices"
INFO_DIR = config.DATA_DIR / "info"
NEWS_DIR = config.DATA_DIR / "news"
FIN_DIR = config.DATA_DIR / "financials"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]

PERIODS = {
    "1M": 31, "3M": 92, "6M": 183, "1Y": 365, "2Y": 730, "5Y": 1826, "10Y": 3652,
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def clean_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def parse_symbols(text: str) -> list[str]:
    seen, out = set(), []
    for s in str(text).replace(";", ",").split(","):
        s = clean_symbol(s)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _safe_name(symbol: str) -> str:
    return symbol.replace("^", "IDX_").replace("/", "_").replace("=", "_")


def _record_source(symbol: str, source: str) -> None:
    try:
        st.session_state.setdefault("_data_sources", {})[symbol] = source
    except Exception:
        pass


def data_sources() -> dict[str, str]:
    try:
        return dict(st.session_state.get("_data_sources", {}))
    except Exception:
        return {}


def reset_sources() -> None:
    try:
        st.session_state["_data_sources"] = {}
    except Exception:
        pass


def online_available() -> bool:
    return (yf is not None) and not config.is_offline()


def _to_date(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def resolve_range(start=None, end=None, period: str | None = None):
    end_ts = _to_date(end) if end is not None else _to_date(dt.date.today())
    if period:
        start_ts = end_ts - pd.Timedelta(days=PERIODS.get(period, 365))
    elif start is not None:
        start_ts = _to_date(start)
    else:
        start_ts = end_ts - pd.Timedelta(days=365)
    if start_ts >= end_ts:
        start_ts = end_ts - pd.Timedelta(days=30)
    return start_ts, end_ts


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV)
    if isinstance(df.columns, pd.MultiIndex):  # yf.download style
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    keep = [c for c in OHLCV if c in df.columns]
    df = df[keep].copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["Close"])
    return df


# --------------------------------------------------------------------------- #
# cache on disk
# --------------------------------------------------------------------------- #
def _price_path(symbol: str):
    return PRICE_DIR / f"{_safe_name(symbol)}.csv"


def load_cached_prices(symbol: str) -> pd.DataFrame:
    path = _price_path(symbol)
    if not path.exists():
        return pd.DataFrame(columns=OHLCV)
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return _normalize_ohlcv(df)
    except Exception as exc:  # corrupted cache should never crash a page
        log.warning("Could not read cache %s: %s", path, exc)
        return pd.DataFrame(columns=OHLCV)


def save_prices(symbol: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    old = load_cached_prices(symbol)
    merged = pd.concat([old, df]) if not old.empty else df
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.to_csv(_price_path(symbol))


def cached_symbols() -> list[str]:
    if not PRICE_DIR.exists():
        return []
    return sorted(p.stem.replace("IDX_", "^") for p in PRICE_DIR.glob("*.csv"))


# --------------------------------------------------------------------------- #
# demo data (synthetic, deterministic per symbol)
# --------------------------------------------------------------------------- #
_DEMO_ORIGIN = pd.Timestamp("2005-01-03")


def _market_factor(n: int) -> np.ndarray:
    rng = np.random.default_rng(20240101)
    return rng.normal(0.0003, 0.011, n)


def demo_prices(symbol: str, start, end) -> pd.DataFrame:
    """Synthetic but realistic-looking OHLCV (geometric Brownian motion with a
    shared market factor so correlations and betas behave sensibly)."""
    symbol = clean_symbol(symbol)
    end_ts = max(_to_date(end), _DEMO_ORIGIN + pd.Timedelta(days=30))
    idx = pd.bdate_range(_DEMO_ORIGIN, end_ts)
    n = len(idx)
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    is_index = symbol.startswith("^")
    beta = 1.0 if is_index else rng.uniform(0.6, 1.6)
    idio_vol = rng.uniform(0.002, 0.004) if is_index else rng.uniform(0.008, 0.022)
    drift = rng.uniform(-0.0001, 0.0004)
    rets = beta * _market_factor(n) + rng.normal(drift, idio_vol, n)
    jumps = rng.random(n) < 0.004  # rare shocks, like earnings surprises
    rets[jumps] += rng.normal(0, 0.06, jumps.sum())
    p0 = rng.uniform(1500, 6000) if is_index else rng.uniform(15, 120)
    close = p0 * np.exp(np.cumsum(rets))
    day_vol = np.abs(rets) + idio_vol
    open_ = close / np.exp(rets) * np.exp(rng.normal(0, idio_vol / 3, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, day_vol / 2)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, day_vol / 2)))
    volume = (rng.lognormal(16.5, 0.35, n) * (1 + 8 * np.abs(rets))).astype("int64")
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=pd.DatetimeIndex(idx, name="Date"),
    )
    start_ts = _to_date(start)
    return df.loc[(df.index >= start_ts) & (df.index <= _to_date(end))]


# --------------------------------------------------------------------------- #
# prices
# --------------------------------------------------------------------------- #
def _download_history(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame(columns=OHLCV)
    raw = yf.Ticker(symbol).history(
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,  # split/dividend adjusted, no separate 'Adj Close'
    )
    return _normalize_ohlcv(raw)


@st.cache_data(ttl=900, show_spinner=False)
def _history_cached(symbol: str, start: str, end: str, offline: bool):
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if not offline and yf is not None:
        try:
            df = _download_history(symbol, start_ts, end_ts)
            if not df.empty:
                try:
                    save_prices(symbol, df)
                except Exception as exc:  # read-only disk on some hosts
                    log.warning("Could not cache %s: %s", symbol, exc)
                return df, "live"
        except Exception as exc:
            log.warning("Live download failed for %s: %s", symbol, exc)
    cached = load_cached_prices(symbol)
    if not cached.empty:
        window = cached.loc[(cached.index >= start_ts) & (cached.index <= end_ts)]
        if len(window) >= 2:
            return window, "cache"
    return demo_prices(symbol, start_ts, end_ts), "demo"


def get_history(symbol: str, start=None, end=None, period: str | None = None) -> pd.DataFrame:
    """Daily OHLCV for one symbol. Never raises; never returns an empty frame
    unless the date range itself is empty."""
    symbol = clean_symbol(symbol)
    start_ts, end_ts = resolve_range(start, end, period)
    df, source = _history_cached(
        symbol, start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"), config.is_offline()
    )
    _record_source(symbol, source)
    return df.copy()


def get_closes(symbols: Iterable[str], start=None, end=None, period: str | None = None,
               dropna: bool = True) -> pd.DataFrame:
    """Close prices for several symbols, one column each, aligned on dates."""
    frames = {}
    for s in symbols:
        df = get_history(s, start, end, period)
        if not df.empty:
            frames[clean_symbol(s)] = df["Close"]
    if not frames:
        return pd.DataFrame()
    closes = pd.DataFrame(frames).sort_index()
    return closes.dropna() if dropna else closes


@st.cache_data(ttl=60, show_spinner=False)
def _intraday_cached(symbol: str, offline: bool):
    if offline or yf is None:
        return None
    try:
        raw = yf.Ticker(symbol).history(period="1d", interval="1m", auto_adjust=True)
        if raw is None or raw.empty:
            raw = yf.Ticker(symbol).history(period="5d", interval="5m", auto_adjust=True)
        if raw is None or raw.empty:
            return None
        raw = raw[[c for c in OHLCV if c in raw.columns]].copy()
        if getattr(raw.index, "tz", None) is not None:
            raw.index = raw.index.tz_localize(None)
        return raw
    except Exception as exc:
        log.warning("Intraday failed for %s: %s", symbol, exc)
        return None


def get_intraday(symbol: str) -> pd.DataFrame | None:
    """Recent intraday bars (online only). Returns None when unavailable."""
    return _intraday_cached(clean_symbol(symbol), config.is_offline())


def get_quote(symbol: str) -> dict:
    """Latest price snapshot. Uses intraday bars online, the last daily bars otherwise."""
    symbol = clean_symbol(symbol)
    daily = get_history(symbol, period="1M")
    source = data_sources().get(symbol, "demo")
    quote = {"symbol": symbol, "source": source}
    if daily.empty:
        return quote
    prev_close = float(daily["Close"].iloc[-2]) if len(daily) > 1 else float(daily["Close"].iloc[-1])
    intraday = get_intraday(symbol)
    if intraday is not None and not intraday.empty:
        last_day = intraday.index[-1].normalize()
        today = intraday.loc[intraday.index.normalize() == last_day]
        quote.update(
            price=float(today["Close"].iloc[-1]), open=float(today["Open"].iloc[0]),
            high=float(today["High"].max()), low=float(today["Low"].min()),
            volume=int(today["Volume"].sum()), time=today.index[-1], source="live",
        )
        if last_day == daily.index[-1] and len(daily) > 1:
            prev_close = float(daily["Close"].iloc[-2])
        else:
            prev_close = float(daily["Close"].iloc[-1])
    else:
        last = daily.iloc[-1]
        quote.update(
            price=float(last["Close"]), open=float(last["Open"]), high=float(last["High"]),
            low=float(last["Low"]), volume=int(last["Volume"]), time=daily.index[-1],
        )
    quote["prev_close"] = prev_close
    quote["change"] = quote["price"] - prev_close
    quote["change_pct"] = quote["change"] / prev_close * 100 if prev_close else 0.0
    return quote


# --------------------------------------------------------------------------- #
# company info, financial statements, news
# --------------------------------------------------------------------------- #
def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path, payload) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, default=str, indent=1), encoding="utf-8")
    except Exception as exc:
        log.warning("Could not write %s: %s", path, exc)


@st.cache_data(ttl=3600, show_spinner=False)
def _info_cached(symbol: str, offline: bool):
    path = INFO_DIR / f"{_safe_name(symbol)}.json"
    if not offline and yf is not None:
        try:
            info = yf.Ticker(symbol).info or {}
            if len(info) > 3:
                _write_json(path, info)
                return info, "live"
        except Exception as exc:
            log.warning("Info failed for %s: %s", symbol, exc)
    cached = _read_json(path) if path.exists() else None
    if cached:
        return cached, "cache"
    return {}, "none"


def get_info(symbol: str) -> tuple[dict, str]:
    """Company profile and key statistics: (info dict, source)."""
    return _info_cached(clean_symbol(symbol), config.is_offline())


@st.cache_data(ttl=3600, show_spinner=False)
def _financials_cached(symbol: str, offline: bool):
    path = FIN_DIR / f"{_safe_name(symbol)}_income.csv"
    if not offline and yf is not None:
        try:
            stmt = yf.Ticker(symbol).income_stmt
            if stmt is not None and not stmt.empty:
                path.parent.mkdir(parents=True, exist_ok=True)
                stmt.to_csv(path)
                return stmt, "live"
        except Exception as exc:
            log.warning("Financials failed for %s: %s", symbol, exc)
    if path.exists():
        try:
            return pd.read_csv(path, index_col=0), "cache"
        except Exception:
            pass
    return pd.DataFrame(), "none"


def get_income_statement(symbol: str) -> tuple[pd.DataFrame, str]:
    return _financials_cached(clean_symbol(symbol), config.is_offline())


def _parse_yf_news(items) -> list[dict]:
    out = []
    for item in items or []:
        content = item.get("content", item)  # new and old yfinance formats
        url = (
            (content.get("canonicalUrl") or {}).get("url")
            or (content.get("clickThroughUrl") or {}).get("url")
            or content.get("link") or ""
        )
        provider = content.get("provider") or {}
        published = content.get("pubDate") or content.get("providerPublishTime") or ""
        if isinstance(published, (int, float)):
            published = dt.datetime.fromtimestamp(published).isoformat()
        out.append({
            "title": content.get("title", ""),
            "summary": content.get("summary") or content.get("description") or "",
            "source": provider.get("displayName") if isinstance(provider, dict) else content.get("publisher", ""),
            "url": url,
            "published": str(published),
            "provider_sentiment": None,
        })
    return [a for a in out if a["title"]]


_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/130.0 Safari/537.36"}


def _alpha_vantage_news(symbol: str, api_key: str) -> list[dict]:
    resp = requests.get(
        "https://www.alphavantage.co/query",
        params={"function": "NEWS_SENTIMENT", "tickers": symbol, "apikey": api_key, "limit": 50},
        timeout=20,
    )
    resp.raise_for_status()
    payload = resp.json()
    if "feed" not in payload:
        # Alpha Vantage reports rate limits and bad keys as a 200 with a message
        msg = payload.get("Information") or payload.get("Note") or payload.get("Error Message") or str(payload)[:200]
        raise RuntimeError(msg)
    out = []
    for a in payload["feed"]:
        score = None
        for ts in a.get("ticker_sentiment", []):
            if ts.get("ticker") == symbol:
                score = float(ts.get("ticker_sentiment_score", 0))
        out.append({
            "title": a.get("title", ""), "summary": a.get("summary", ""),
            "source": a.get("source", ""), "url": a.get("url", ""),
            "published": a.get("time_published", ""),
            "provider_sentiment": score if score is not None else a.get("overall_sentiment_score"),
        })
    return [a for a in out if a["title"]]


def _yfinance_news(symbol: str) -> list[dict]:
    if yf is None:
        raise RuntimeError("yfinance is not installed")
    t = yf.Ticker(symbol)
    try:
        items = t.get_news(count=30, tab="news")
    except (AttributeError, TypeError):  # older yfinance
        items = t.news
    return _parse_yf_news(items)


def _rss_news(url: str, source_name: str) -> list[dict]:
    import html
    import re
    import xml.etree.ElementTree as ET

    resp = requests.get(url, headers=_UA, timeout=20)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    out = []
    for item in root.iter("item"):
        title = html.unescape(item.findtext("title") or "").strip()
        desc = re.sub(r"<[^>]+>", " ", html.unescape(item.findtext("description") or ""))
        src = item.find("source")
        publisher = src.text if src is not None and src.text else source_name
        if source_name == "Google News" and " - " in title:  # "Headline - Publisher"
            title, publisher = title.rsplit(" - ", 1)
            desc = ""  # Google's description only repeats the title
        out.append({
            "title": title, "summary": re.sub(r"\s+", " ", desc).strip()[:500], "source": publisher,
            "url": item.findtext("link") or "", "published": item.findtext("pubDate") or "",
            "provider_sentiment": None,
        })
    return [a for a in out if a["title"]]


class _NoNews(Exception):
    """Raised inside the cached function so failures are not cached."""


@st.cache_data(ttl=900, show_spinner=False)
def _news_live(symbol: str, av_key: str | None):
    from urllib.parse import quote_plus

    providers = []
    if av_key:
        providers.append(("Alpha Vantage", lambda: _alpha_vantage_news(symbol, av_key)))
    providers += [
        ("Yahoo Finance", lambda: _yfinance_news(symbol)),
        ("Yahoo Finance RSS", lambda: _rss_news(
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote_plus(symbol)}&region=US&lang=en-US",
            "Yahoo Finance")),
        ("Google News", lambda: _rss_news(
            f"https://news.google.com/rss/search?q={quote_plus(symbol + ' stock')}&hl=en-US&gl=US&ceid=US:en",
            "Google News")),
    ]
    errors = {}
    for name, fetch in providers:
        try:
            articles = fetch()
            if articles:
                return articles, name, errors
            errors[name] = "returned no headlines"
        except Exception as exc:  # each provider is independent
            errors[name] = f"{type(exc).__name__}: {exc}"[:300]
            log.warning("News provider %s failed for %s: %s", name, symbol, exc)
    raise _NoNews(errors)


def get_news(symbol: str) -> tuple[list[dict], str]:
    """Headlines from the first provider that works; cached to data/news for offline use.
    Provider errors are kept in st.session_state['_news_errors'] for troubleshooting."""
    symbol = clean_symbol(symbol)
    path = NEWS_DIR / f"{_safe_name(symbol)}.json"
    errors = {}
    if not config.is_offline():
        try:
            articles, source, errors = _news_live(symbol, config.get_secret("ALPHA_VANTAGE_API_KEY") or None)
            _write_json(path, {"source": source, "articles": articles})
            _set_news_errors(errors)
            return articles, source
        except _NoNews as exc:
            errors = exc.args[0]
    _set_news_errors(errors)
    cached = _read_json(path) if path.exists() else None
    if cached:
        return cached.get("articles", []), f"cache ({cached.get('source', '?')})"
    return [], "none"


def _set_news_errors(errors: dict) -> None:
    try:
        st.session_state["_news_errors"] = dict(errors)
    except Exception:
        pass


def news_errors() -> dict:
    try:
        return dict(st.session_state.get("_news_errors", {}))
    except Exception:
        return {}
