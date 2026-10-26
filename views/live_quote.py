import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core import data, ui

ui.header("Live Quote", "Latest price, today's range and an intraday chart that can refresh itself.")

c1, c2 = st.columns([2, 1])
with c1:
    symbol = ui.symbol_input("Ticker", "AAPL", key="lq_symbol")
auto = c2.toggle("Auto-refresh every 30 s", value=False, disabled=not data.online_available())


@st.fragment(run_every=30 if auto else None)
def quote_panel(sym: str) -> None:
    q = data.get_quote(sym)
    if "price" not in q:
        st.error(f"No data for {sym}.")
        return
    m = st.columns(5)
    m[0].metric("Price", ui.money(q["price"]), f"{q['change']:+.2f} ({q['change_pct']:+.2f}%)")
    m[1].metric("Open", ui.money(q["open"]))
    m[2].metric("Day high", ui.money(q["high"]))
    m[3].metric("Day low", ui.money(q["low"]))
    m[4].metric("Volume", ui.big_number(q["volume"]))
    label = {"live": "Live", "cache": "Cached", "demo": "Demo"}.get(q["source"], q["source"])
    st.caption(f"{label} data · as of {q['time']:%Y-%m-%d %H:%M}")

    intraday = data.get_intraday(sym)
    if intraday is not None and not intraday.empty:
        last_day = intraday.index[-1].normalize()
        bars = intraday.loc[intraday.index.normalize() == last_day]
        title = f"{sym} intraday · {last_day:%d %b %Y}"
    else:
        bars = data.get_history(sym, period="3M")
        title = f"{sym} · last 3 months (daily)"

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    color = ui.UP if bars["Close"].iloc[-1] >= bars["Close"].iloc[0] else ui.DOWN
    fig.add_trace(go.Scatter(x=bars.index, y=bars["Close"], mode="lines", name="Price",
                             line=dict(color=color, width=2), fill="tozeroy",
                             fillcolor="rgba(56,189,248,0.06)"), row=1, col=1)
    fig.add_hline(y=q["prev_close"], line_dash="dot", line_color="#94A3B8",
                  annotation_text="prev close", row=1, col=1)
    fig.add_trace(go.Bar(x=bars.index, y=bars["Volume"], name="Volume", marker_color="#475569"), row=2, col=1)
    fig.update_yaxes(range=[bars["Low"].min() * 0.998, bars["High"].max() * 1.002], row=1, col=1)
    ui.chart(fig, height=520, title=title)
    ui.data_status()


quote_panel(symbol)
if not data.online_available():
    st.caption("Offline: showing the most recent cached daily bar instead of live intraday prices.")
