"""Provider-agnostic LLM helper (Groq or Gemini). Picks a working model by asking the API what exists."""
import os
import time

GROQ_PREFERRED = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-20b"]
GEMINI_PREFERRED = ["gemini-3.8-flash", "gemini-3.8-flash-lite", "gemini-2.5-flash"]
_cache = {}


def detect_provider(api_key):
    forced = os.getenv("LLM_PROVIDER", "").lower()
    if forced in ("groq", "gemini"):
        return forced
    return "groq" if api_key.startswith("gsk_") else "gemini"


def _retry(fn):
    last = None
    for attempt in range(5):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if any(c in str(e) for c in ("429", "500", "502", "503", "504", "overloaded", "UNAVAILABLE")):
                last = e
                time.sleep(2 * (attempt + 1))
            else:
                raise
    raise last


def _pick(available, preferred, ok):
    for p in preferred:
        if p in available:
            return p
    rest = [m for m in available if ok(m)]
    if not rest:
        raise RuntimeError(f"No usable model found. Available: {available[:10]}")
    return rest[0]


def call_groq(prompt, api_key):
    from groq import Groq
    client = Groq(api_key=api_key)
    if "m" not in _cache:
        ids = [m.id for m in client.models.list().data]
        _cache["m"] = _pick(ids, GROQ_PREFERRED,
                            lambda m: "llama" in m and not any(x in m for x in ("guard", "whisper", "tts")))
    model = _cache["m"]
    kwargs = dict(model=model, temperature=0, max_completion_tokens=4096,
                  messages=[{"role": "system", "content": "You reply with valid JSON only, no extra text."},
                            {"role": "user", "content": prompt}])
    if "gpt-oss" in model:                      # reasoning model: keep thinking short so it doesn't eat the output
        kwargs["reasoning_effort"] = "low"
    # 1st try: strict JSON mode. If the model/provider rejects it or returns nothing, retry in plain mode
    # (our parser extracts the JSON from the text anyway).
    for json_mode in (True, False):
        try:
            extra = {"response_format": {"type": "json_object"}} if json_mode else {}
            r = _retry(lambda: client.chat.completions.create(**kwargs, **extra))
            text = (r.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as e:  # noqa: BLE001
            if json_mode and ("json_validate_failed" in str(e) or "400" in str(e)):
                continue
            raise
    raise RuntimeError("The model returned an empty reply")


def call_gemini(prompt, api_key):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    if "m" not in _cache:
        ids = [m.name.replace("models/", "") for m in client.models.list()]
        _cache["m"] = _pick(ids, GEMINI_PREFERRED, lambda m: "flash" in m and "image" not in m)
    r = _retry(lambda: client.models.generate_content(
        model=_cache["m"], contents=prompt,
        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json")))
    return r.text


def call_llm(prompt, api_key):
    return call_groq(prompt, api_key) if detect_provider(api_key) == "groq" else call_gemini(prompt, api_key)


def used_model():
    return _cache.get("m", "unknown")
