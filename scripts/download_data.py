"""Pre-fill the offline cache (data/) so the app works without internet.

Usage:
    python scripts/download_data.py                  # default watchlist, 10 years
    python scripts/download_data.py AAPL TSLA SAP.DE --years 5
    python scripts/download_data.py --no-info        # prices only (faster)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yfinance as yf  # noqa: E402

from core import config  # noqa: E402
from core.data import FIN_DIR, INFO_DIR, NEWS_DIR, _normalize_ohlcv, _parse_yf_news, _safe_name, save_prices  # noqa: E402

DEFAULT = (config.DEFAULT_SYMBOLS + ["AMD", "NFLX", "JPM", "KO"]
           + list(config.MARKET_INDICES.values()) + list(config.SECTOR_ETFS.values()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbols", nargs="*", help="Tickers to cache (default: built-in watchlist)")
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--no-info", action="store_true", help="Skip company info, financials and news")
    args = ap.parse_args()

    symbols = [s.upper() for s in args.symbols] or DEFAULT
    start = (dt.date.today() - dt.timedelta(days=365 * args.years)).isoformat()
    ok, failed = 0, []
    for i, sym in enumerate(symbols, 1):
        try:
            t = yf.Ticker(sym)
            df = _normalize_ohlcv(t.history(start=start, auto_adjust=True))
            if df.empty:
                raise ValueError("no price data")
            save_prices(sym, df)
            msg = f"{len(df)} days"
            if not args.no_info and not sym.startswith("^"):
                try:
                    info = t.info or {}
                    if info:
                        INFO_DIR.mkdir(parents=True, exist_ok=True)
                        (INFO_DIR / f"{_safe_name(sym)}.json").write_text(json.dumps(info, default=str))
                    stmt = t.income_stmt
                    if stmt is not None and not stmt.empty:
                        FIN_DIR.mkdir(parents=True, exist_ok=True)
                        stmt.to_csv(FIN_DIR / f"{_safe_name(sym)}_income.csv")
                    news = _parse_yf_news(t.news)
                    if news:
                        NEWS_DIR.mkdir(parents=True, exist_ok=True)
                        (NEWS_DIR / f"{_safe_name(sym)}.json").write_text(
                            json.dumps({"source": "Yahoo Finance", "articles": news}, default=str))
                    msg += ", info, financials, news"
                except Exception as exc:  # fundamentals are best-effort
                    msg += f" (info skipped: {exc})"
            print(f"[{i}/{len(symbols)}] {sym}: {msg}")
            ok += 1
        except Exception as exc:
            print(f"[{i}/{len(symbols)}] {sym}: FAILED ({exc})")
            failed.append(sym)
        time.sleep(0.3)  # be gentle with Yahoo
    print(f"\nCached {ok}/{len(symbols)} symbols in {config.DATA_DIR}")
    if failed:
        print("Failed:", ", ".join(failed))


if __name__ == "__main__":
    main()
