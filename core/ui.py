"""Shared UI building blocks so every page looks and behaves the same."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import config, data

ACCENT = "#38BDF8"
UP = "#22C55E"
DOWN = "#EF4444"
PALETTE = ["#38BDF8", "#F59E0B", "#A78BFA", "#22C55E", "#F472B6", "#EF4444", "#2DD4BF", "#FACC15"]


def header(title: str, subtitle: str = "") -> None:
    data.reset_sources()
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def symbol_input(label: str = "Ticker", default: str = "AAPL", key: str | None = None,
                 sidebar: bool = False) -> str:
    box = st.sidebar if sidebar else st
    value = box.text_input(label, default, key=key, help="Yahoo Finance symbol, e.g. AAPL, MSFT, ^GSPC, SAP.DE")
    return data.clean_symbol(value) or default


def symbols_input(label: str = "Tickers (comma-separated)", default: str = "AAPL, MSFT, GOOGL",
                  key: str | None = None, sidebar: bool = False, min_count: int = 1) -> list[str]:
    box = st.sidebar if sidebar else st
    syms = data.parse_symbols(box.text_input(label, default, key=key))
    if len(syms) < min_count:
        st.warning(f"Enter at least {min_count} ticker(s).")
        st.stop()
    return syms


def date_range(default_years: float = 2, key: str = "dr", sidebar: bool = False):
    box = st.sidebar if sidebar else st
    today = dt.date.today()
    default_start = today - dt.timedelta(days=int(365 * default_years))
    c1, c2 = (box.columns(2) if not sidebar else (st.sidebar, st.sidebar))
    start = c1.date_input("Start date", default_start, max_value=today, key=f"{key}_start")
    end = c2.date_input("End date", today, max_value=today, key=f"{key}_end")
    if start >= end:
        st.error("Start date must be before end date.")
        st.stop()
    return start, end


def period_picker(default: str = "1Y", key: str = "period", options=None) -> str:
    options = options or list(data.PERIODS)
    return st.segmented_control("Period", options, default=default, key=key) or default


def data_status() -> None:
    """Tell the user when they are looking at cached or demo data."""
    sources = data.data_sources()
    demo = sorted(s for s, v in sources.items() if v == "demo")
    cache = sorted(s for s, v in sources.items() if v == "cache")
    if demo:
        st.warning(
            f"**Demo data** for {', '.join(demo)}: these prices are synthetic, generated "
            "so the page works without internet. Go online once (or run "
            "`python scripts/download_data.py`) to cache real prices.",
            icon=":material/science:",
        )
    if cache:
        st.info(f"Offline: showing **cached** prices for {', '.join(cache)}.", icon=":material/cloud_off:")


def require_online(feature: str) -> bool:
    if data.online_available():
        return True
    st.info(f"{feature} needs an internet connection. Turn off **Offline mode** in the sidebar.",
            icon=":material/cloud_off:")
    return False


def style(fig: go.Figure, height: int = 420, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        colorway=PALETTE,
        hovermode="x unified",
        title=title,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.15)")
    return fig


def chart(fig: go.Figure, height: int = 420, title: str | None = None) -> None:
    st.plotly_chart(style(fig, height, title), width="stretch")


# ----------------------------------------------------------------------------
# formatting
# ----------------------------------------------------------------------------
def money(x, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"${x:,.{decimals}f}"


def pct(x, decimals: int = 2, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{decimals}%}" if sign else f"{x:.{decimals}%}"


def big_number(x) -> str:
    if x is None:
        return "n/a"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "n/a"
    for unit, size in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= size:
            return f"{x / size:,.2f}{unit}"
    return f"{x:,.2f}"


def disclaimer() -> None:
    st.caption("Educational project. Nothing here is investment advice.")


def to_csv_download(df: pd.DataFrame, filename: str, label: str = "Download CSV") -> None:
    st.download_button(label, df.to_csv().encode("utf-8"), file_name=filename, mime="text/csv",
                       icon=":material/download:")
