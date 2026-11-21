# PS66 - Financial News Summarizer

**Problem statement:** Investors miss important news. This tool fetches financial headlines from an RSS feed, summarises each in one line with an LLM and tags it Positive / Negative / Neutral for the company, shown as a colour-coded list.

## Features
- Choose a source (Economic Times, Moneycontrol, Livemint, CNBC, Yahoo Finance), a custom RSS URL, or offline sample headlines
- LLM returns a 1-line summary, the company affected and a sentiment tag for every headline
- Colour-coded cards (green = positive, red = negative, grey = neutral), sentiment counts and filter
- Batching, JSON validation, retries and automatic model selection (Groq or Gemini)
- Download the results as CSV

## How to run
```
pip install -r requirements.txt
cp .env.example .env      # put your free Groq API key in .env
streamlit run app.py
```
Free key: https://console.groq.com/keys

## Files
- `app.py` - Streamlit UI
- `summarizer.py` - RSS fetching (feedparser), prompt, parsing, validation
- `llm.py` - Groq/Gemini client with retries and model discovery
- `sample_headlines.csv` - offline test headlines
- `test_summarizer.py` - tests with a stub LLM

## Screenshots
Add 1-2 screenshots here.
