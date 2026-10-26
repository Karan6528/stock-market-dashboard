import re

import streamlit as st

from core import config, ui

ui.header("YouTube Videos", "Recent videos about a stock, with a quick extractive summary of each transcript.")

api_key = config.get_secret("YOUTUBE_API_KEY")
if not api_key:
    st.info(
        "This page needs a **YouTube Data API v3** key. Create one in Google Cloud Console, then add it to "
        "`.streamlit/secrets.toml`:\n\n```toml\nYOUTUBE_API_KEY = \"your-key\"\n```",
        icon=":material/key:",
    )
    st.stop()
if not ui.require_online("Video search"):
    st.stop()

try:
    from googleapiclient.discovery import build
except ImportError:
    st.error("Install the optional dependency: `pip install google-api-python-client youtube-transcript-api`")
    st.stop()


@st.cache_data(ttl=3600, show_spinner=False)
def search(query: str, key: str, n: int):
    yt = build("youtube", "v3", developerKey=key, cache_discovery=False)
    resp = yt.search().list(q=query, part="snippet", type="video", maxResults=n, order="date").execute()
    return [{
        "title": i["snippet"]["title"], "channel": i["snippet"]["channelTitle"],
        "published": i["snippet"]["publishedAt"][:10], "id": i["id"]["videoId"],
    } for i in resp.get("items", [])]


@st.cache_data(ttl=86400, show_spinner=False)
def summary(video_id: str, n_sentences: int = 4) -> str:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        try:  # v1.x API
            parts = YouTubeTranscriptApi().fetch(video_id)
            text = " ".join(p.text for p in parts)
        except AttributeError:  # older versions
            text = " ".join(p["text"] for p in YouTubeTranscriptApi.get_transcript(video_id))
    except Exception:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) < 3:  # auto captions often have no punctuation
        words = text.split()
        return " ".join(words[:80]) + ("…" if len(words) > 80 else "")
    return " ".join(sentences[:n_sentences])


c1, c2 = st.columns([3, 1])
query = c1.text_input("Search", "NVIDIA stock analysis")
n = c2.number_input("Videos", 1, 10, 4)
if query:
    try:
        videos = search(query, api_key, int(n))
    except Exception as exc:
        st.error(f"YouTube search failed: {exc}")
        st.stop()
    if not videos:
        st.warning("No videos found.")
    for v in videos:
        with st.container(border=True):
            c1, c2 = st.columns([1.3, 2])
            c1.video(f"https://www.youtube.com/watch?v={v['id']}")
            c2.markdown(f"**{v['title']}**")
            c2.caption(f"{v['channel']} · {v['published']}")
            s = summary(v["id"])
            c2.write(s if s else "_Transcript not available for this video._")
