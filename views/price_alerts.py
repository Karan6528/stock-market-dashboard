import datetime as dt
import smtplib
from email.message import EmailMessage

import pandas as pd
import streamlit as st

from core import config, data, ui

ui.header("Price Alerts", "Get notified when a stock crosses a price or moves more than a set percentage.")

if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []

smtp_ready = all(config.get_secret(k) for k in ("SMTP_USER", "SMTP_PASSWORD"))

with st.form("new_alert", clear_on_submit=True):
    c1, c2, c3, c4 = st.columns([1, 1.4, 1, 1.4])
    sym = c1.text_input("Ticker", "AAPL")
    kind = c2.selectbox("Condition", ["Price above", "Price below", "Daily move ≥ % (either way)"])
    level = c3.number_input("Level", 0.0, 1e6, 200.0)
    email = c4.text_input("Email (optional)", disabled=not smtp_ready,
                          placeholder="you@example.com" if smtp_ready else "set SMTP secrets to enable")
    if st.form_submit_button("Add alert", icon=":material/add_alert:"):
        st.session_state.alerts.append({"symbol": data.clean_symbol(sym), "kind": kind, "level": level,
                                        "email": email.strip(), "active": True})


def send_email(to: str, subject: str, body: str) -> bool:
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, config.get_secret("SMTP_USER"), to
    msg.set_content(body)
    try:
        host = config.get_secret("SMTP_HOST", "smtp.gmail.com")
        port = int(config.get_secret("SMTP_PORT", 465))
        with smtplib.SMTP_SSL(host, port, timeout=15) as server:
            server.login(config.get_secret("SMTP_USER"), config.get_secret("SMTP_PASSWORD"))
            server.send_message(msg)
        return True
    except Exception:
        return False


if not st.session_state.alerts:
    st.info("No alerts yet. Add one above.")
    st.stop()

interval = st.select_slider("Check every", [30, 60, 120, 300], value=60, format_func=lambda s: f"{s} s")
running = st.toggle("Monitoring on", value=data.online_available(), disabled=not data.online_available())
if not data.online_available():
    st.caption(":material/cloud_off: Alerts check live prices, so they're paused in offline mode. "
               "The rules below are evaluated once against the latest cached prices.")


@st.fragment(run_every=interval if running else None)
def monitor():
    rows = []
    for a in st.session_state.alerts:
        q = data.get_quote(a["symbol"])
        price, chg = q.get("price"), q.get("change_pct", 0.0)
        if price is None:
            rows.append({"Ticker": a["symbol"], "Rule": f"{a['kind']} {a['level']:g}", "Price": None,
                         "Today %": None, "Status": "no data"})
            continue
        if a["kind"] == "Price above":
            hit = price >= a["level"]
        elif a["kind"] == "Price below":
            hit = price <= a["level"]
        else:
            hit = abs(chg) >= a["level"]
        status = "TRIGGERED" if hit else "watching"
        if hit and a["active"]:
            a["active"] = False  # fire once
            text = f"{a['symbol']} {a['kind'].lower()} {a['level']}: price {price:.2f} ({chg:+.2f}% today)"
            st.session_state.alert_log.insert(0, {"time": dt.datetime.now().strftime("%H:%M:%S"), "alert": text})
            st.toast(text, icon=":material/notifications_active:")
            if a["email"] and smtp_ready:
                ok = send_email(a["email"], f"Stock alert: {a['symbol']}", text)
                st.session_state.alert_log[0]["alert"] += " · email sent" if ok else " · email failed"
        rows.append({"Ticker": a["symbol"], "Rule": f"{a['kind']} {a['level']:g}", "Price": price,
                     "Today %": chg, "Status": status if a["active"] or hit else "fired"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Price": st.column_config.NumberColumn(format="$%.2f"),
        "Today %": st.column_config.NumberColumn(format="%.2f%%"),
    })
    st.caption(f"Last checked {dt.datetime.now():%H:%M:%S}")
    if st.session_state.alert_log:
        st.markdown("**Alert log**")
        st.dataframe(pd.DataFrame(st.session_state.alert_log), hide_index=True, width="stretch")


monitor()
c1, c2 = st.columns(2)
if c1.button("Re-arm all alerts"):
    for a in st.session_state.alerts:
        a["active"] = True
    st.rerun()
if c2.button("Clear alerts"):
    st.session_state.alerts, st.session_state.alert_log = [], []
    st.rerun()
st.caption("Alerts run while this page is open in your browser. For 24/7 alerts you'd run a scheduled job "
           "(cron, GitHub Actions) instead of a web app.")
