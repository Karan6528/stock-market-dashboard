"""App-wide settings: paths, secrets and the offline switch."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
MARKET_INDICES = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
}
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Consumer Staples": "XLP",
}
TRADING_DAYS = 252


def get_secret(name: str, default=None):
    """Read a setting from .streamlit/secrets.toml, then environment variables.

    API keys never live in the code. Put them in .streamlit/secrets.toml locally
    or in the app's Secrets settings when deployed.
    """
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # no secrets file at all is fine
        pass
    return os.environ.get(name, default)


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def default_offline() -> bool:
    return _truthy(get_secret("OFFLINE_MODE", "false"))


def is_offline() -> bool:
    """True when the user switched the app to offline mode (cache/demo data only)."""
    if "offline_mode" in st.session_state:
        return bool(st.session_state["offline_mode"])
    return default_offline()
