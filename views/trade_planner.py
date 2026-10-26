import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import data, indicators as ind, planner, sentiment, ui

ui.header("Trade Planner",
          "Enter your budget and the planner works out how many shares to buy, where to put a stop-loss and a "
          "target, and how many days to hold, based on how this stock has actually moved in the past.")

c1, c2, c3 = st.columns(3)
with c1:
    symbol = ui.symbol_input("Ticker", "AAPL", key="tp_symbol")
capital = c2.number_input("Money to invest ($)", 100.0, 100_000_000.0, 5_000.0, step=500.0)
risk_pct = c3.slider("Max loss you accept (% of that money)", 0.5, 20.0, 5.0, 0.5,
                     help="If the stop-loss is hit, you lose at most this share of your budget.") / 100
style = st.segmented_control("Trading style", list(planner.STYLES), default=list(planner.STYLES)[0], key="tp_style") \
    or list(planner.STYLES)[0]
stop_atr, rr, max_days = planner.STYLES[style]
with st.expander("Advanced settings", icon=":material/tune:"):
    a, b, c, d = st.columns(4)
    stop_atr = a.number_input("Stop distance (× ATR)", 0.5, 10.0, float(stop_atr), 0.5,
                              help="ATR = average daily price range. 2 × ATR leaves room for normal noise.")
    rr = b.number_input("Reward : risk", 0.5, 10.0, float(rr), 0.5, help="Target distance as a multiple of the stop distance.")
    max_days = int(c.number_input("Max holding days", 5, 500, int(max_days), 5))
    fractional = d.toggle("Fractional shares", False)

df = data.get_history(symbol, period="10Y")
if len(df) < 260:
    st.error("Need at least a year of price history for this ticker.")
    st.stop()

price = float(df["Close"].iloc[-1])
atr = float(ind.atr(df).iloc[-1])
plan = planner.position_size(capital, price, atr, risk_pct, stop_atr, rr, fractional)
stop_pct = plan.risk_per_share / price
target_pct = plan.target / price - 1

if plan.shares <= 0:
    st.error(f"Your budget can't buy even one share at {ui.money(price)} within the risk limit. Increase the "
             "budget or allowed loss, or turn on fractional shares.")
    st.stop()

# ------------------------------------------------------------------- history
outcomes = planner.historical_outcomes(df, target_pct, stop_pct, max_days)
stats = planner.summarise(outcomes, stop_pct)

articles, _ = data.get_news(symbol)
news_score = (np.mean([sentiment.score(f"{a['title']}. {a.get('summary', '')}") for a in articles[:20]])
              if articles else None)
checks = planner.setup_checks(df, news_score)
passed = [ok for _, ok, _ in checks if ok is not None]
score = sum(passed) / len(passed) if passed else 0.5
edge = stats.get("expected_return", 0)
if score >= 0.75 and edge > 0:
    light, headline = "🟢", "Setup looks favourable by these rules"
elif score <= 0.4 or edge < 0:
    light, headline = "🔴", "Setup looks unfavourable by these rules"
else:
    light, headline = "🟡", "Mixed signals: be selective"

# ---------------------------------------------------------------------- plan
hold_days = stats.get("median_days_target")
time_stop = stats.get("p80_days_target")
hold_txt = f"about {hold_days:.0f} trading days" if pd.notna(hold_days) else "n/a"
with st.container(border=True):
    c1, c2 = st.columns([1, 5])
    c1.markdown(f"<div style='font-size:72px;text-align:center'>{light}</div>", unsafe_allow_html=True)
    c2.markdown(f"### Your plan for {symbol}")
    c2.markdown(
        f"Buy **{plan.shares:g} share{'' if plan.shares == 1 else 's'}** at about **{ui.money(price)}** (≈ {ui.money(plan.invested)}), set a "
        f"**stop-loss at {ui.money(plan.stop)}** and a **target at {ui.money(plan.target)}**. "
        f"Historically, the target was reached in **{hold_txt}** (median). "
        + (f"If it hasn't happened within **{time_stop:.0f} days**, consider closing the position (time stop)."
           if pd.notna(time_stop) else "")
    )
    c2.caption(f"{headline}. Position size limited by your {plan.limited_by}.")

m = st.columns(4)
m[0].metric("Shares to buy", f"{plan.shares:g}", f"{ui.money(plan.invested, 0)} invested", delta_color="off")
m[1].metric("Stop-loss", ui.money(plan.stop), f"-{stop_pct:.1%}", delta_color="inverse")
m[2].metric("Target", ui.money(plan.target), f"+{target_pct:.1%}")
m[3].metric("Suggested holding", f"{hold_days:.0f} days" if pd.notna(hold_days) else "n/a",
            f"time stop {time_stop:.0f} days" if pd.notna(time_stop) else None, delta_color="off")
m = st.columns(4)
m[0].metric("Max loss if stopped", ui.money(plan.max_loss), ui.pct(plan.max_loss / capital, 1), delta_color="off")
m[1].metric("Gain if target hit", ui.money(plan.potential_gain), ui.pct(plan.potential_gain / capital, 1),
            delta_color="off")
m[2].metric("Cash left over", ui.money(plan.cash_left))
m[3].metric("Daily range (ATR 14)", ui.money(atr), ui.pct(atr / price, 1), delta_color="off")

# ------------------------------------------------------------------- chart
recent = df.tail(180)
future = pd.bdate_range(recent.index[-1], periods=max_days + 1)
fig = go.Figure()
fig.add_trace(go.Candlestick(x=recent.index, open=recent["Open"], high=recent["High"], low=recent["Low"],
                             close=recent["Close"], name=symbol, increasing_line_color=ui.UP,
                             decreasing_line_color=ui.DOWN))
for lvl, name, color in ((plan.target, "Target", ui.UP), (price, "Entry", ui.ACCENT), (plan.stop, "Stop-loss", ui.DOWN)):
    fig.add_trace(go.Scatter(x=[recent.index[0], future[-1]], y=[lvl, lvl], mode="lines", name=f"{name} {lvl:.2f}",
                             line=dict(color=color, dash="dash", width=1.5)))
fig.add_vrect(x0=future[0], x1=future[min(int(time_stop) if pd.notna(time_stop) else max_days, max_days)],
              fillcolor=ui.ACCENT, opacity=0.08, line_width=0, annotation_text="holding window")
fig.update_layout(xaxis_rangeslider_visible=False)
ui.chart(fig, 460, "Entry, stop-loss and target")

# ------------------------------------------------------------- backtest of plan
st.subheader("🕰️ What happened when this plan was used in the past?")
if stats:
    st.caption(f"The same plan (target +{target_pct:.1%}, stop −{stop_pct:.1%}, max {max_days} days) was replayed "
               f"from every trading day in the last {len(df) / 252:.0f} years ({stats['n']:,} starting points).")
    c = st.columns(4)
    c[0].metric("🎯 Target reached first", ui.pct(stats["p_target"], 0))
    c[1].metric("🛑 Stop hit first", ui.pct(stats["p_stop"], 0))
    c[2].metric("⏱️ Neither (time limit)", ui.pct(stats["p_time"], 0))
    c[3].metric("Average result per trade", ui.pct(stats["expected_return"], 2, sign=True),
                f"{stats['expected_R']:+.2f} R", delta_color="off",
                help="R = the amount risked. +0.3 R means you earned 30% of what you risked, on average.")
    wins = outcomes[outcomes["Outcome"] == "Target"]
    if len(wins):
        hist = go.Figure(go.Histogram(x=wins["Days"], nbinsx=min(40, max_days), marker_color=ui.UP))
        hist.add_vline(x=hold_days, line_dash="dash", annotation_text=f"median {hold_days:.0f}d")
        hist.add_vline(x=time_stop, line_dash="dot", annotation_text=f"80% within {time_stop:.0f}d",
                       annotation_position="bottom right")
        hist.update_layout(xaxis_title="Trading days until target was hit", yaxis_title="Times",
                           hovermode="closest")
        ui.chart(hist, 320, "How long did it take to reach the target?")
    breakeven = 1 / (1 + rr)
    st.caption(f"With a {rr:g}:1 reward/risk you only need to win more than {breakeven:.0%} of trades to break even. "
               "History shows how this stock behaved, not how it will behave.")

# --------------------------------------------------------------- checklist
st.subheader("✅ Setup checklist")
for name, ok, detail in checks:
    icon = "✅" if ok else "⚪" if ok is None else "❌"
    st.markdown(f"{icon} **{name}** · {detail}")
if news_score is None:
    st.caption("No news cached for this ticker, so news tone isn't included. Open News Sentiment once while online.")

with st.expander("How the numbers are calculated"):
    st.markdown(f"""
- **Stop-loss** = price − {stop_atr:g} × ATR(14). ATR is the stock's average daily range, so the stop sits
  outside normal day-to-day noise.
- **Shares** = (budget × max loss %) ÷ (price − stop), capped by the cash you have. This is standard
  *fixed-fractional* position sizing: a bigger daily range means a wider stop, so you buy fewer shares.
- **Target** = price + {rr:g} × (price − stop).
- **Holding period**: the plan was replayed from every past trading day. The *suggested holding* is the
  median number of days it took to reach the target, and the *time stop* is the point by which 80% of
  successful trades had reached it.
""")
ui.data_status()
st.caption("⚠️ Educational tool, not financial advice. It doesn't know your finances, taxes or goals. "
           "Never invest money you can't afford to lose.")
