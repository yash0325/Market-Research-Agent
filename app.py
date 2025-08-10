# Streamlit Cloud-ready Market Research App
# - Reads OPENAI_API_KEY from Streamlit Secrets (secrets.toml)
# - No dotenv required
# - Paste multiple URLs; runs extract → sentiment → trends → exec summary

import os
import json
import time
import asyncio
from typing import List, Dict, Any, Optional

import streamlit as st

# LLM / LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# HTTP / Parsing
import requests
from bs4 import BeautifulSoup
from readability import Document

# Utils
import pandas as pd

# ---- Configure OpenAI key from Streamlit Secrets ----
# In Streamlit Community Cloud, set in your app's Secrets as:
# [general]
# OPENAI_API_KEY = "sk-..."
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ---------------- Prompts ----------------
CONTENT_EXTRACTOR_TMPL = """You are a content extraction agent.
Extract the most important factual information, statistics, and statements from the source below.
Ignore fluff, ads, unrelated commentary. Return concise bullets.

Source:
{chunk}

Key facts:"""

SENTIMENT_TMPL = """You are a sentiment analysis agent.
Given the extracted content below, assess overall sentiment (positive, negative, neutral) with 2–3 bullets of reasoning.
Note if sentiment varies by topic.

Extracted:
{extracted}

Sentiment:"""

TREND_TMPL = """You are a trend analysis agent.
Identify recurring themes, trends, new products, noteworthy changes. Note frequency or cross-source emphasis where possible.

Extracted:
{extracted}

Trends:"""

REPORT_TMPL = """You are an executive report writer.
Write a concise summary (<= 300 words) for a business executive. Focus on actionable insights, opportunities, and risks.
Start with 3–5 bullet headlines, then 1 short paragraph.

Extracted:
{extracted}

Sentiment:
{sentiment}

Trends:
{trends}

Executive Summary:"""

# -------------- Helpers --------------
def build_llm(model_name: str = "gpt-4o", temperature: float = 0.2) -> ChatOpenAI:
    # ChatOpenAI will read OPENAI_API_KEY from environment (set above from st.secrets)
    return ChatOpenAI(model=model_name, temperature=temperature)

def chunk_texts(records: List[Dict[str, Any]], chunk_size: int = 3000, chunk_overlap: int = 300):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for rec in records:
        for i, ch in enumerate(splitter.split_text(rec["text"])):
            chunks.append({"url": rec["url"], "title": rec["title"], "chunk": ch, "idx": i})
    return chunks

def _clean_urls(raw: str) -> List[str]:
    urls = [u.strip() for u in raw.splitlines() if u.strip()]
    seen, uniq = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq

def _simple_bs_extract(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    article_like = soup.find_all(["article", "main"]) or []
    text_blocks = [blk.get_text(" ", strip=True) for blk in article_like] or [soup.get_text(" ", strip=True)]
    text = "\n".join([t for t in text_blocks if t])
    return f"{title}\n{text}".strip()

async def _fetch_one(url: str, timeout_sec: int = 25) -> Optional[Dict[str, Any]]:
    def try_newspaper():
        try:
            from newspaper import Article
            art = Article(url)
            art.download(); art.parse()
            title = art.title or ""
            text = f"{title}\n{art.text}".strip()
            publish_date = None
            if getattr(art, "publish_date", None):
                try: publish_date = art.publish_date.isoformat()
                except Exception: pass
            return {"url": url, "title": title, "text": text, "publish_date": publish_date} if text.strip() else None
        except Exception:
            return None

    def try_readability():
        try:
            resp = requests.get(url, timeout=timeout_sec)
            resp.raise_for_status()
            doc = Document(resp.text)
            title = doc.short_title() or ""
            text = _simple_bs_extract(doc.summary())
            return {"url": url, "title": title, "text": f"{title}\n{text}".strip(), "publish_date": None} if text.strip() else None
        except Exception:
            return None

    def try_bs4():
        try:
            resp = requests.get(url, timeout=timeout_sec)
            resp.raise_for_status()
            text = _simple_bs_extract(resp.text)
            title = text.split("\n", 1)[0] if "\n" in text else ""
            return {"url": url, "title": title, "text": text, "publish_date": None} if text.strip() else None
        except Exception:
            return None

    for fn in (try_newspaper, try_readability, try_bs4):
        item = fn()
        if item and len(item.get("text", "")) > 500:
            return item
    return None

async def _fetch_many(urls: List[str]) -> List[Dict[str, Any]]:
    tasks = [_fetch_one(u) for u in urls]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r]

@st.cache_data(show_spinner=False)
def fetch_articles(urls: List[str]) -> List[Dict[str, Any]]:
    return asyncio.run(_fetch_many(urls))

def map_reduce_extract(llm: ChatOpenAI, chunks: List[Dict[str, Any]]) -> str:
    extract_prompt = PromptTemplate.from_template(CONTENT_EXTRACTOR_TMPL)
    extract_chain = extract_prompt | llm | StrOutputParser()

    bullets = []
    for ch in chunks:
        out = extract_chain.invoke({"chunk": ch["chunk"]})
        bullets.append(out)

    reducer_prompt = PromptTemplate.from_template(
        "Deduplicate and merge the following bullet points into a single concise list of key facts:\n\n{bullets}\n\nMerged key facts:"
    )
    reducer_chain = reducer_prompt | llm | StrOutputParser()
    merged = reducer_chain.invoke({"bullets": "\n".join(bullets)})
    return merged

def run_agents(llm: ChatOpenAI, extracted: str) -> Dict[str, str]:
    sentiment_prompt = PromptTemplate.from_template(SENTIMENT_TMPL)
    trend_prompt = PromptTemplate.from_template(TREND_TMPL)
    report_prompt = PromptTemplate.from_template(REPORT_TMPL)

    sentiment = (sentiment_prompt | llm | StrOutputParser()).invoke({"extracted": extracted})
    trends = (trend_prompt | llm | StrOutputParser()).invoke({"extracted": extracted})
    summary = (report_prompt | llm | StrOutputParser()).invoke(
        {"extracted": extracted, "sentiment": sentiment, "trends": trends}
    )
    return {"sentiment": sentiment, "trends": trends, "summary": summary}

# ---------------- UI ----------------
st.set_page_config(page_title="Market Research (Streamlit Cloud)", page_icon="🕵️", layout="wide")
st.title("🕵️ Market Research — Streamlit Cloud Ready")
st.caption("Paste URLs or upload a list, run extraction → sentiment → trends → summary, and capture metrics.")

with st.sidebar:
    st.header("Model & Settings")
    model_name = st.selectbox("OpenAI model", ["gpt-4o", "gpt-4o-mini", "o4-mini"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    chunk_size = st.number_input("Chunk size", min_value=1000, max_value=8000, value=3000, step=250)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=2000, value=300, step=50)

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set in Secrets. Add it in Settings → Secrets.")

st.subheader("Enter or Upload URLs")
urls_text = st.text_area("One per line", height=120, placeholder="https://example.com/article-1\nhttps://example.com/blog-post")
upload = st.file_uploader("…or upload a urls.txt", type=["txt"])

if upload is not None:
    uploaded_text = upload.read().decode("utf-8", errors="ignore")
    urls_text = (urls_text + "\n" + uploaded_text).strip()

col1, col2, col3 = st.columns([1,1,1])
with col1:
    run_btn = st.button("Run Analysis", type="primary")
with col2:
    sample_btn = st.button("Load Sample URLs")
with col3:
    clear_btn = st.button("Clear")

if sample_btn:
    st.session_state["urls_text"] = "https://www.bbc.com/news\nhttps://www.cnn.com"
    st.rerun()
if clear_btn:
    st.session_state["urls_text"] = ""
    st.rerun()
if "urls_text" in st.session_state and not urls_text.strip():
    urls_text = st.session_state["urls_text"]

if run_btn:
    start_time = time.time()
    urls = _clean_urls(urls_text)
    if not urls:
        st.warning("Please enter at least one URL.")
        st.stop()
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set in Secrets.")
        st.stop()

    with st.status("Fetching articles…", expanded=True) as status:
        st.write(f"Fetching {len(urls)} URL(s)…")
        fetch_t0 = time.time()
        try:
            articles = fetch_articles(urls)
        except Exception as e:
            st.error(f"Fetch failed: {e}")
            st.stop()
        fetch_t1 = time.time()
        if not articles:
            st.error("No articles could be fetched.")
            st.stop()
        st.success(f"Fetched {len(articles)} / {len(urls)}")
        status.update(label="Fetched", state="complete")

    with st.expander("Preview fetched sources", expanded=False):
        for a in articles:
            st.markdown(f"**Title:** {a['title'] or '(no title)'}  \n**URL:** {a['url']}")
            st.write((a["text"][:800] + "…") if len(a["text"]) > 800 else a["text"])
            st.divider()

    llm = build_llm(model_name, temperature)

    st.info("Chunking…")
    chunks = chunk_texts(articles, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
    st.write(f"Created {len(chunks)} chunk(s).")

    with st.status("Extracting key facts (map‑reduce)…", expanded=False):
        try:
            x_t0 = time.time()
            extracted = map_reduce_extract(llm, chunks)
            x_t1 = time.time()
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            st.stop()
        st.success("Extraction complete.")

    with st.expander("Key Extracted Content", expanded=True):
        st.write(extracted)

    with st.status("Sentiment, trends, summary…", expanded=False):
        try:
            a_t0 = time.time()
            results = run_agents(llm, extracted)
            a_t1 = time.time()
        except Exception as e:
            st.error(f"Agent runs failed: {e}")
            st.stop()
        st.success("Analysis complete.")

    # -------- Metrics panel --------
    total_time = time.time() - start_time
    fetch_time = fetch_t1 - fetch_t0
    extract_time = x_t1 - x_t0
    agents_time = a_t1 - a_t0
    per_article_minutes = (total_time / max(1, len(articles))) / 60.0

    st.subheader("📊 Run Metrics")
    mcols = st.columns(3)
    with mcols[0]:
        st.metric("Articles fetched", f"{len(articles)}/{len(urls)}")
        st.metric("Avg minutes/article", f"{per_article_minutes:.2f}")
    with mcols[1]:
        st.metric("Fetch time (s)", f"{fetch_time:.1f}")
        st.metric("Extract time (s)", f"{extract_time:.1f}")
    with mcols[2]:
        st.metric("Agents time (s)", f"{agents_time:.1f}")

    # Per‑URL mini‑summaries
    st.divider()
    st.subheader("Per‑URL Mini‑Summaries")
    mini_summaries = []
    mini_prompt = PromptTemplate.from_template(
        "Summarize in 3 bullets (facts only). Source text:\n\n{chunk}\n\nBullets:")
    mini_chain = mini_prompt | llm | StrOutputParser()
    for rec in articles:
        snippet = rec["text"][:1500]
        bullets = mini_chain.invoke({"chunk": snippet})
        mini_summaries.append({"url": rec["url"], "title": rec["title"], "bullets": bullets})
        with st.expander(f"{rec['title'] or '(no title)'} — {rec['url']}"):
            st.write(bullets)

    # Downloads
    st.divider(); st.subheader("Download Artifacts")
    files = {
        "sources.jsonl": "\n".join(json.dumps(a, ensure_ascii=False) for a in articles).encode("utf-8"),
        "extracted_key_facts.txt": extracted.strip().encode("utf-8"),
        "sentiment.txt": results["sentiment"].strip().encode("utf-8"),
        "trends.txt": results["trends"].strip().encode("utf-8"),
        "executive_summary.txt": results["summary"].strip().encode("utf-8"),
        "mini_summaries.json": json.dumps(mini_summaries, ensure_ascii=False, indent=2).encode("utf-8"),
    }

    d1, d2, d3, d4, d5 = st.columns(5)
    with d1: st.download_button("sources.jsonl", data=files["sources.jsonl"], file_name="sources.jsonl")
    with d2: st.download_button("extracted_key_facts.txt", data=files["extracted_key_facts.txt"], file_name="extracted_key_facts.txt")
    with d3: st.download_button("sentiment.txt", data=files["sentiment.txt"], file_name="sentiment.txt")
    with d4: st.download_button("trends.txt", data=files["trends.txt"], file_name="trends.txt")
    with d5: st.download_button("executive_summary.txt", data=files["executive_summary.txt"], file_name="executive_summary.txt")

# Footer
st.caption("Tip: Add your OPENAI_API_KEY in Settings → Secrets. Paste multiple URLs — one per line.")
