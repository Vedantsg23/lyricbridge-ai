"""LyricBridge AI: meaning-first lyric translation (a Vercel Python function).

GET  /api/translate   -> which LLM the server can use, if any
POST /api/translate   {"language": "Spanish", "lines": ["...", "..."]}
                      -> {"lines": [{"en": "...", "note": "..."}], "provider": "...", "model": "..."}

Only the lyric text is sent here; audio never leaves the visitor's browser.
Standard library only, so the function needs no install step.

Credentials, first match wins (LLM_PROVIDER picks one explicitly, LLM_MODEL overrides the model):
  GEMINI_API_KEY      Google Gemini API
  OPENAI_API_KEY      OpenAI
  GROQ_API_KEY        Groq
  AI_GATEWAY_API_KEY  Vercel AI Gateway (on Vercel the deployment's OIDC token also works)
Set LLM_DISABLED=1 to switch the endpoint off.
"""
import json
import os
import re
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler

MAX_LINES = 80
MAX_CHARS = 8000
MAX_BODY = 64_000
TIMEOUT = 45

SYSTEM_PROMPT = """You translate song lyrics into natural English that keeps their meaning.
Translate the meaning, imagery and feeling of each line, not word by word.
Rules:
- Return exactly one English line for every input line, in the same order.
- If a line is already in English, return it unchanged.
- Add a short note (18 words at most) only when a line contains an idiom, slang, wordplay or a cultural
  reference whose meaning would otherwise be lost. Otherwise leave the note empty.
- Do not add, merge or drop lines, and write nothing outside the JSON.
Answer with JSON only, in this shape: {"lines": [{"en": "...", "note": ""}]}"""

PROVIDERS = {
    "gemini": {"key_env": "GEMINI_API_KEY", "model": "gemini-flash-lite-latest"},
    "openai": {"key_env": "OPENAI_API_KEY", "model": "gpt-4.1-mini", "base": "https://api.openai.com/v1"},
    "groq": {"key_env": "GROQ_API_KEY", "model": "llama-3.3-70b-versatile", "base": "https://api.groq.com/openai/v1"},
    "gateway": {"key_env": "AI_GATEWAY_API_KEY", "model": "google/gemini-2.5-flash-lite",
                "base": "https://ai-gateway.vercel.sh/v1"},
}
ORDER = ("gemini", "openai", "groq", "gateway")


class ProviderError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def choose_provider(env, headers=None):
    """Return (name, key, model) for the LLM to use, or None when nothing is configured."""
    if env.get("LLM_DISABLED"):
        return None
    found = []
    for name in ORDER:
        key = env.get(PROVIDERS[name]["key_env"])
        if name == "gateway" and not key:
            key = (headers.get("x-vercel-oidc-token") if headers else None) or env.get("VERCEL_OIDC_TOKEN")
        if key:
            found.append((name, key))
    wanted = (env.get("LLM_PROVIDER") or "").strip().lower()
    if wanted:
        found = [f for f in found if f[0] == wanted]
    if not found:
        return None
    name, key = found[0]
    return name, key, (env.get("LLM_MODEL") or PROVIDERS[name]["model"])


def build_user_prompt(language, lines):
    return json.dumps({"source_language": language or "unknown", "lines": lines}, ensure_ascii=False)


def parse_reply(text, count):
    """Pull the {"lines": [...]} object out of a model reply and return exactly `count` items."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in the reply")
    data = json.loads(cleaned[start:end + 1])
    items = data.get("lines") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError("reply has no 'lines' list")
    out = []
    for item in items[:count]:
        if isinstance(item, str):
            out.append({"en": item.strip(), "note": ""})
        elif isinstance(item, dict):
            out.append({"en": str(item.get("en") or "").strip(), "note": str(item.get("note") or "").strip()[:200]})
        else:
            out.append({"en": "", "note": ""})
    out.extend({"en": "", "note": ""} for _ in range(count - len(out)))
    return out


def _post_json(url, headers, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", "replace")
        try:
            body = json.loads(detail)
            detail = (body.get("error") or {}).get("message") if isinstance(body.get("error"), dict) else body.get("error") or detail
        except (ValueError, AttributeError):
            pass
        raise ProviderError(err.code, str(detail)[:300]) from None
    except urllib.error.URLError as err:
        raise ProviderError(502, f"could not reach the model: {err.reason}") from None


def call_provider(name, key, model, system, user):
    if name == "gemini":
        data = _post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            {"x-goog-api-key": key},
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0.3, "responseMimeType": "application/json"},
            },
        )
        parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts)
    data = _post_json(
        PROVIDERS[name]["base"] + "/chat/completions",
        {"Authorization": f"Bearer {key}"},
        {
            "model": model,
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        },
    )
    return data["choices"][0]["message"]["content"]


def validate(body):
    """Return (language, lines) or raise ValueError with a message for the visitor."""
    if not isinstance(body, dict):
        raise ValueError("Send a JSON object with a 'lines' list.")
    lines = body.get("lines")
    if not isinstance(lines, list) or not lines or not all(isinstance(x, str) for x in lines):
        raise ValueError("'lines' must be a non-empty list of strings.")
    if len(lines) > MAX_LINES:
        raise ValueError(f"Send at most {MAX_LINES} lines per request.")
    if sum(len(x) for x in lines) > MAX_CHARS:
        raise ValueError(f"Send at most {MAX_CHARS} characters per request.")
    language = str(body.get("language") or "")[:40]
    return language, [x.strip()[:400] for x in lines]


class handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        chosen = choose_provider(os.environ, self.headers)
        self._send(200, {
            "llm": bool(chosen),
            "provider": chosen[0] if chosen else None,
            "model": chosen[2] if chosen else None,
            "max_lines": MAX_LINES,
        })

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        if length > MAX_BODY:
            return self._send(413, {"error": "Request too large."})
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._send(400, {"error": "Send valid JSON."})
        try:
            language, lines = validate(body)
        except ValueError as err:
            return self._send(400, {"error": str(err)})
        chosen = choose_provider(os.environ, self.headers)
        if not chosen:
            return self._send(503, {"error": "No LLM is configured on this server."})
        name, key, model = chosen
        try:
            reply = call_provider(name, key, model, SYSTEM_PROMPT, build_user_prompt(language, lines))
            result = parse_reply(reply, len(lines))
        except ProviderError as err:
            return self._send(502, {"error": f"{name} answered {err.status}: {err.message}", "provider": name})
        except (ValueError, KeyError, IndexError, TypeError):
            return self._send(502, {"error": "The model did not return valid JSON.", "provider": name})
        self._send(200, {"lines": result, "provider": name, "model": model})

    def log_message(self, *args):  # keep function logs quiet
        pass
