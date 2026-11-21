"""PS66 - Financial News Summarizer (Streamlit UI)."""
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from llm import used_model
from summarizer import FEEDS, fetch_news, summarize_news

load_dotenv()
st.set_page_config(page_title="Financial News Summarizer", layout="wide")
st.title("Financial News Summarizer")
st.caption("Fetches headlines from an RSS feed. The LLM gives each a 1-line summary and a Positive / Negative tag.")

api_key = (os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")
           or st.sidebar.text_input("API key (Groq gsk_... or Gemini)", type="password"))
source = st.sidebar.selectbox("News source", list(FEEDS) + ["Custom RSS URL", "Sample headlines (offline)"])
custom = st.sidebar.text_input("Custom RSS URL") if source == "Custom RSS URL" else ""
limit = st.sidebar.slider("Number of headlines", 5, 20, 10)

COLORS = {"Positive": ("#d4edda", "#155724", "🟢"), "Negative": ("#f8d7da", "#721c24", "🔴"),
          "Neutral": ("#e2e3e5", "#383d41", "⚪")}

if st.button("Fetch & summarise", type="primary"):
    if not api_key:
        st.error("Enter an API key in the sidebar (or set GROQ_API_KEY in .env).")
        st.stop()
    try:
        if source == "Sample headlines (offline)":
            news = pd.read_csv("sample_headlines.csv").head(limit).to_dict("records")
        else:
            news = fetch_news(custom or FEEDS[source], limit)
    except Exception as e:  # noqa: BLE001
        st.error(f"{e}. Try another source, or choose 'Sample headlines (offline)'.")
        st.stop()
    if not news:
        st.warning("The feed returned no items.")
        st.stop()
    bar = st.progress(0.0)
    st.session_state["news"] = summarize_news(news, api_key, progress=bar.progress)
    bar.empty()
    st.success(f"Done. {len(news)} headlines summarised. Model used: {used_model()}")

if "news" in st.session_state:
    items = st.session_state["news"]
    df = pd.DataFrame(items)
    counts = df["sentiment"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Positive", int(counts.get("Positive", 0)))
    c2.metric("🔴 Negative", int(counts.get("Negative", 0)))
    c3.metric("⚪ Neutral", int(counts.get("Neutral", 0)))
    show = st.multiselect("Show sentiment", list(COLORS), default=list(COLORS))
    for n in items:
        if n["sentiment"] not in show:
            continue
        bg, fg, icon = COLORS[n["sentiment"]]
        link = f' <a href="{n["link"]}" target="_blank">read more</a>' if n.get("link") else ""
        st.markdown(
            f'<div style="background:{bg};color:{fg};padding:12px 16px;border-radius:8px;margin-bottom:8px;">'
            f'<b>{icon} {n["sentiment"]}</b> &nbsp;|&nbsp; <b>{n["company"]}</b><br>'
            f'{n["summary"]}<br><small><i>{n["title"]}</i>{link}</small></div>', unsafe_allow_html=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "financial_news_summary.csv", "text/csv")
