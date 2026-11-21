"""Core logic for PS66 - Financial News Summarizer: RSS -> LLM (1-line summary + sentiment) -> results."""
import json
import re

import certifi
import feedparser
import requests

from llm import call_llm

FEEDS = {
    "Economic Times - Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Moneycontrol - Business": "https://www.moneycontrol.com/rss/business.xml",
    "Livemint - Markets": "https://www.livemint.com/rss/markets",
    "CNBC - Finance": "https://www.cnbc.com/id/10000664/device/rss/rss.xml",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
}
SENTIMENTS = ["Positive", "Negative", "Neutral"]

PROMPT = """You are a financial news analyst helping busy investors.
For each news item below do three things:
1. summary: ONE sentence, max 25 words, plain English, keep key numbers.
2. company: the main company or stock affected (or "Market" if it is general).
3. sentiment: how this news is for that company/market: exactly Positive, Negative or Neutral.
   (Positive = good for share price, e.g. profit growth, big order, upgrade.
    Negative = bad, e.g. loss, fraud, fine, downgrade, layoffs. Neutral = no clear impact.)

Return ONLY a JSON object, one entry per item:
{{"results": [{{"id": 0, "summary": "...", "company": "...", "sentiment": "..."}}]}}

News items:
{items}
"""


def strip_html(text):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


def fetch_news(url, limit=10):
    """Read an RSS feed. Returns list of dicts: title, description, link, published."""
    # Download with certifi's CA bundle (fixes SSL errors on macOS python.org installs), then parse.
    try:
        resp = requests.get(url, timeout=15, verify=certifi.where(),
                            headers={"User-Agent": "Mozilla/5.0 (FinancialNewsSummarizer)"})
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Could not download feed: {e}") from e
    feed = feedparser.parse(resp.content)
    if feed.bozo and not feed.entries:
        raise RuntimeError(f"Could not read feed: {getattr(feed, 'bozo_exception', 'unknown error')}")
    out = []
    for e in feed.entries[:limit]:
        out.append({"title": strip_html(e.get("title", "")),
                    "description": strip_html(e.get("summary", ""))[:300],
                    "link": e.get("link", ""),
                    "published": e.get("published", "")})
    return out


def build_prompt(batch):
    items = "\n".join(f"{i}. {n['title']} - {n['description']}" if n.get("description")
                      else f"{i}. {n['title']}" for i, n in batch)
    return PROMPT.format(items=items)


def parse_response(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"[\{\[].*[\}\]]", text, flags=re.DOTALL)   # grab the JSON even if the model added extra words
    data = json.loads(m.group(0) if m else text)
    if isinstance(data, dict):
        data = data["results"] if "results" in data else [data]
    return data


def _norm(r):
    s = next((x for x in SENTIMENTS if x.lower() == str(r.get("sentiment", "")).lower()), "Neutral")
    return {"id": int(r["id"]), "summary": str(r.get("summary", "")).strip(),
            "company": str(r.get("company", "Market")).strip() or "Market", "sentiment": s}


def summarize_news(news, api_key, batch_size=8, llm=call_llm, progress=None):
    """Return news list with summary / company / sentiment added to each item."""
    res = {}

    def run(batch):
        """Ask the LLM about one batch; return True if parsed OK."""
        for _ in range(2):
            try:
                for r in parse_response(llm(build_prompt(batch), api_key)):
                    n = _norm(r)
                    res[n["id"]] = n
                return True
            except (json.JSONDecodeError, KeyError, ValueError, TypeError, RuntimeError):
                continue
        return False

    for start in range(0, len(news), batch_size):
        batch = list(enumerate(news[start:start + batch_size], start=start))
        if not run(batch) and len(batch) > 1:      # batch failed: retry each headline on its own
            for item in batch:
                if item[0] not in res:
                    run([item])
        if progress:
            progress(min(start + batch_size, len(news)) / len(news))
    out = []
    for i, n in enumerate(news):
        r = res.get(i, {"summary": "(could not summarise)", "company": "Market", "sentiment": "Neutral"})
        out.append({**n, "summary": r["summary"], "company": r["company"], "sentiment": r["sentiment"]})
    return out
