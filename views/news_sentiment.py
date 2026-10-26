from collections import Counter
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import data, sentiment, ui

ui.header("News Sentiment",
          f"Each headline is scored with NLP ({sentiment.engine()}) and labelled 😊 good, 😐 neutral or 😟 bad, "
          "then checked against how the stock price actually reacted.")

tab_news, tab_text = st.tabs(["📰 Ticker news", "✍️ Analyze your own text"])

with tab_news:
    symbol = ui.symbol_input("Ticker", "AAPL", key="news_symbol")
    articles, source = data.get_news(symbol)
    if not articles:
        if not data.online_available():
            st.info("No cached headlines for this ticker. Go online once to fetch and cache them.",
                    icon=":material/cloud_off:")
        else:
            st.warning("No headlines could be fetched from any news source. Details below.")
        errs = data.news_errors()
        if errs:
            with st.expander("Why? (provider errors)", expanded=True):
                for name, err in errs.items():
                    st.markdown(f"- **{name}**: `{err}`")
                st.caption("Alpha Vantage's free key allows 25 requests a day. Yahoo sometimes rate-limits "
                           "(HTTP 429): wait a minute and press R to rerun.")
        st.stop()

    prices = data.get_history(symbol, period="3M")["Close"]
    rows = []
    for a in articles[:30]:
        text = f"{a['title']}. {a.get('summary', '')}"
        pol = sentiment.score(text)
        published = sentiment.parse_published(a.get("published"))
        day, r1, r3 = sentiment.price_reaction(published, prices)
        rows.append({**a, "score": pol, "label": sentiment.label(pol), "emoji": sentiment.emoji(pol),
                     "published_ts": published, "reaction_day": day, "ret_1d": r1, "ret_3d": r3})
    df = pd.DataFrame(rows).sort_values("published_ts", ascending=False, na_position="last")

    # ------------------------------------------------------------------ verdict
    avg = df["score"].mean()
    counts = df["label"].value_counts()
    big = sentiment.verdict(avg)
    with st.container(border=True):
        c1, c2 = st.columns([1, 3])
        c1.markdown(f"<div style='font-size:76px;text-align:center;line-height:1.1'>{sentiment.emoji(avg)}</div>",
                    unsafe_allow_html=True)
        c2.markdown(f"### Overall: {big}")
        c2.markdown(f"Across **{len(df)} headlines** about **{symbol}**: "
                    f"😊 {int(counts.get('Positive', 0))} good · 😐 {int(counts.get('Neutral', 0))} neutral · "
                    f"😟 {int(counts.get('Negative', 0))} bad. Average score **{avg:+.2f}** (−1 to +1).")
        c2.caption(f"Source: {source}")

    # ------------------------------------------------------- market reaction
    st.subheader("📈 Did the market agree?")
    judged = df.dropna(subset=["ret_1d"])
    judged = judged[judged["label"] != "Neutral"]
    if len(judged):
        agree = (np.sign(judged["ret_1d"]) == np.where(judged["label"] == "Positive", 1, -1))
        m = st.columns(3)
        m[0].metric("Headlines with a price reaction", len(judged))
        m[1].metric("Market moved in the predicted direction", ui.pct(agree.mean(), 0))
        avg_good = judged.loc[judged["label"] == "Positive", "ret_1d"].mean()
        avg_bad = judged.loc[judged["label"] == "Negative", "ret_1d"].mean()
        m[2].metric("Avg move: 😊 vs 😟 news",
                    f"{(avg_good if pd.notna(avg_good) else 0):+.2%} / {(avg_bad if pd.notna(avg_bad) else 0):+.2%}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices.index, y=prices, name=symbol, line=dict(color="#CBD5E1", width=1.6)))
        for lab, color in (("Positive", ui.UP), ("Negative", ui.DOWN), ("Neutral", "#94A3B8")):
            sel = df[(df["label"] == lab) & df["reaction_day"].notna()]
            if sel.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sel["reaction_day"], y=prices.reindex(sel["reaction_day"]).values, mode="markers+text",
                text=sel["emoji"], textposition="top center", textfont=dict(size=18),
                marker=dict(size=9, color=color), name=sentiment.VERDICT[lab],
                customdata=np.c_[sel["title"].str.slice(0, 80), sel["ret_1d"] * 100],
                hovertemplate="%{customdata[0]}<br>Next-session move %{customdata[1]:.2f}%<extra></extra>",
            ))
        fig.update_layout(hovermode="closest", yaxis_title="Price ($)")
        ui.chart(fig, 420, "Headlines plotted on the day the market could react")
        st.caption("A headline published after the 16:00 New York close is matched to the next trading day. "
                   "Prices move for many reasons at once, so a single headline rarely explains a whole day's move.")
    else:
        st.info("None of these headlines have a trading session after them yet (or they're all neutral), so "
                "there's no price reaction to compare.")

    # -------------------------------------------------------------- charts
    c1, c2 = st.columns([1, 2])
    colors = {"Positive": ui.UP, "Neutral": "#94A3B8", "Negative": ui.DOWN}
    donut = go.Figure(go.Pie(labels=[f"{sentiment.EMOJI[k]} {k}" for k in counts.index], values=counts.values,
                             hole=0.6, marker_colors=[colors[k] for k in counts.index]))
    donut.update_layout(showlegend=True, hovermode="closest")
    with c1:
        ui.chart(donut, 300, "Breakdown")
    hist = px.histogram(df, x="score", nbins=15, color="label", color_discrete_map=colors)
    hist.update_layout(xaxis_title="Sentiment score (−1 to +1)", yaxis_title="Headlines", bargap=0.05)
    with c2:
        ui.chart(hist, 300, "Score distribution")

    words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", " ".join(df["title"]).lower())
    stop = {"with", "from", "that", "this", "will", "into", "after", "about", "over", "stock", "stocks", "shares",
            "says", "what", "your", "their", "more", "than", "have", "just", "here", "why", "could", "should",
            symbol.lower()}
    top = Counter(w for w in words if w not in stop).most_common(15)
    if top:
        with st.expander("Most frequent words in headlines"):
            tw = pd.DataFrame(top, columns=["word", "count"]).sort_values("count")
            bar = go.Figure(go.Bar(x=tw["count"], y=tw["word"], orientation="h", marker_color=ui.ACCENT))
            bar.update_layout(hovermode="closest")
            ui.chart(bar, 380)

    # ------------------------------------------------------------ headlines
    st.subheader("📰 Headlines")
    flt = st.segmented_control("Show", ["All", "😊 Good", "😐 Neutral", "😟 Bad"], default="All", key="news_filter")
    pick = {"😊 Good": "Positive", "😐 Neutral": "Neutral", "😟 Bad": "Negative"}.get(flt or "All")
    shown = df if pick is None else df[df["label"] == pick]
    for _, a in shown.iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([1, 9])
            c1.markdown(f"<div style='font-size:40px;text-align:center'>{a['emoji']}</div>", unsafe_allow_html=True)
            title = f"[{a['title']}]({a['url']})" if a.get("url") else a["title"]
            c2.markdown(f"**{title}**")
            when = f"{a['published_ts']:%d %b %Y %H:%M}" if pd.notna(a["published_ts"]) else str(a.get("published", ""))[:16]
            parts = [a.get("source") or "Unknown", when,
                     f"{sentiment.VERDICT[a['label']]} (score {a['score']:+.2f})"]
            if pd.notna(a.get("provider_sentiment")):
                parts.append(f"Alpha Vantage score {float(a['provider_sentiment']):+.2f}")
            if pd.notna(a["ret_1d"]):
                hit = (a["label"] == "Positive" and a["ret_1d"] > 0) or (a["label"] == "Negative" and a["ret_1d"] < 0)
                mark = "✅" if hit else "➖" if a["label"] == "Neutral" else "❌"
                r3 = f", 3 days {a['ret_3d']:+.2%}" if pd.notna(a["ret_3d"]) else ""
                parts.append(f"{mark} price {a['ret_1d']:+.2%} next session{r3}")
            c2.caption(" · ".join(parts))
            if a.get("summary"):
                c2.write(a["summary"][:400] + ("…" if len(a["summary"]) > 400 else ""))

with tab_text:
    text = st.text_area("Paste a headline, tweet or paragraph",
                        "Apple beats earnings expectations as iPhone sales surge to a record quarter")
    if text.strip():
        pol = sentiment.score(text)
        c1, c2 = st.columns([1, 4])
        c1.markdown(f"<div style='font-size:72px;text-align:center'>{sentiment.emoji(pol)}</div>",
                    unsafe_allow_html=True)
        c2.metric("Verdict", sentiment.VERDICT[sentiment.label(pol)], f"score {pol:+.2f}", delta_color="off")
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) > 1:
            st.dataframe(pd.DataFrame({
                "": [sentiment.emoji(sentiment.score(s)) for s in sentences],
                "Sentence": sentences,
                "Score": [round(sentiment.score(s), 2) for s in sentences],
            }), hide_index=True, width="stretch")
    st.caption("Sentiment of text is not a trading signal by itself. Markets often price news in before you read it.")
