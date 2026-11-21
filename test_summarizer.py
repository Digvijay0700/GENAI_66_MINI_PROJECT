import json
import pandas as pd
from summarizer import parse_response, summarize_news, strip_html


def fake_llm(prompt, key):
    ids = [int(l.split(".")[0]) for l in prompt.split("News items:")[1].strip().splitlines() if l[:1].isdigit()]
    return "```json\n" + json.dumps({"results": [{"id": i, "summary": "s", "company": "X",
                                                  "sentiment": "positive"} for i in ids]}) + "\n```"


def test_parse_fenced_object():
    assert parse_response('```json\n{"results":[{"id":1}]}\n```') == [{"id": 1}]


def test_strip_html():
    assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"


def test_summarize_batches_and_normalises():
    news = pd.read_csv("sample_headlines.csv").fillna("").to_dict("records")
    out = summarize_news(news, "k", batch_size=4, llm=fake_llm)
    assert len(out) == 10 and {n["sentiment"] for n in out} == {"Positive"}


def test_bad_json_falls_back():
    out = summarize_news([{"title": "a", "description": ""}], "k", llm=lambda p, k: "nope")
    assert out[0]["sentiment"] == "Neutral" and "could not" in out[0]["summary"]


def test_parse_extracts_json_from_chatter():
    assert parse_response('Sure! Here you go: {"results":[{"id":2}]} Hope it helps') == [{"id": 2}]


def test_falls_back_to_single_items():
    calls = []

    def flaky(prompt, key):                      # fails for multi-item batches, works for single items
        ids = [int(l.split(".")[0]) for l in prompt.split("News items:")[1].strip().splitlines() if l[:1].isdigit()]
        calls.append(len(ids))
        if len(ids) > 1:
            raise RuntimeError("empty")
        return json.dumps({"results": [{"id": ids[0], "summary": "ok", "company": "X", "sentiment": "Negative"}]})

    out = summarize_news([{"title": "a", "description": ""}, {"title": "b", "description": ""}], "k", llm=flaky)
    assert [n["summary"] for n in out] == ["ok", "ok"] and 1 in calls


def test_groq_retries_without_json_mode(monkeypatch):
    import sys, types, llm
    seen = []

    class Comp:
        def create(self, **kw):
            seen.append("response_format" in kw)
            if "response_format" in kw:
                raise Exception("Error code: 400 - json_validate_failed")
            msg = types.SimpleNamespace(content='{"results": []}')
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = types.SimpleNamespace(completions=Comp())
            self.models = types.SimpleNamespace(list=lambda: types.SimpleNamespace(
                data=[types.SimpleNamespace(id="openai/gpt-oss-20b")]))

    monkeypatch.setitem(sys.modules, "groq", types.SimpleNamespace(Groq=FakeGroq))
    llm._cache.clear()
    assert llm.call_groq("p", "gsk_x") == '{"results": []}' and seen == [True, False]
