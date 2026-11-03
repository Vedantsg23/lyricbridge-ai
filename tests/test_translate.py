"""Tests for api/translate.py. Run with: python3 -m unittest discover -s tests -p 'test_*.py'"""
import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
import translate  # noqa: E402


class ChooseProvider(unittest.TestCase):
    def test_nothing_configured(self):
        self.assertIsNone(translate.choose_provider({}, {}))

    def test_order_and_model_override(self):
        env = {"GROQ_API_KEY": "g", "OPENAI_API_KEY": "o"}
        self.assertEqual(translate.choose_provider(env), ("openai", "o", "gpt-4.1-mini"))
        env["LLM_MODEL"] = "custom"
        self.assertEqual(translate.choose_provider(env)[2], "custom")

    def test_explicit_provider(self):
        env = {"GEMINI_API_KEY": "k", "GROQ_API_KEY": "g", "LLM_PROVIDER": "groq"}
        self.assertEqual(translate.choose_provider(env)[0], "groq")
        env["LLM_PROVIDER"] = "openai"
        self.assertIsNone(translate.choose_provider(env))

    def test_gateway_from_oidc_header(self):
        chosen = translate.choose_provider({}, {"x-vercel-oidc-token": "tok"})
        self.assertEqual(chosen[:2], ("gateway", "tok"))

    def test_disabled(self):
        self.assertIsNone(translate.choose_provider({"LLM_DISABLED": "1", "GEMINI_API_KEY": "k"}))


class ParseReply(unittest.TestCase):
    def test_fenced_json_and_padding(self):
        reply = '```json\n{"lines": [{"en": "Under the moon", "note": ""}]}\n```'
        self.assertEqual(translate.parse_reply(reply, 2),
                         [{"en": "Under the moon", "note": ""}, {"en": "", "note": ""}])

    def test_extra_text_strings_and_truncation(self):
        reply = 'Sure! {"lines": ["one", {"en": "two", "note": "idiom"}, "three"]} Hope it helps.'
        self.assertEqual(translate.parse_reply(reply, 2),
                         [{"en": "one", "note": ""}, {"en": "two", "note": "idiom"}])

    def test_garbage(self):
        with self.assertRaises(ValueError):
            translate.parse_reply("no json here", 1)
        with self.assertRaises(ValueError):
            translate.parse_reply('{"other": 1}', 1)


class Validate(unittest.TestCase):
    def test_limits(self):
        with self.assertRaises(ValueError):
            translate.validate({"lines": []})
        with self.assertRaises(ValueError):
            translate.validate({"lines": ["x"] * (translate.MAX_LINES + 1)})
        with self.assertRaises(ValueError):
            translate.validate({"lines": ["x" * 300] * 30})
        self.assertEqual(translate.validate({"lines": [" hola "], "language": "Spanish"}), ("Spanish", ["hola"]))


class Endpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), translate.handler)
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/api/translate"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        self.saved_env = dict(os.environ)
        for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "AI_GATEWAY_API_KEY", "VERCEL_OIDC_TOKEN", "LLM_PROVIDER"):
            os.environ.pop(k, None)
        self.saved_call = translate.call_provider

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.saved_env)
        translate.call_provider = self.saved_call

    def request(self, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_status_without_llm(self):
        self.assertEqual(self.request(), (200, {"llm": False, "provider": None, "model": None, "max_lines": 80}))

    def test_post_without_llm(self):
        status, body = self.request({"lines": ["hola"]})
        self.assertEqual(status, 503)

    def test_bad_request(self):
        status, body = self.request({"lines": "hola"})
        self.assertEqual(status, 400)
        self.assertIn("lines", body["error"])

    def test_translation(self):
        os.environ["GEMINI_API_KEY"] = "test-key"
        seen = {}

        def fake(name, key, model, system, user):
            seen.update(name=name, key=key, user=json.loads(user))
            return '{"lines": [{"en": "You are my other half", "note": "Literally \'my half orange\'."}]}'

        translate.call_provider = fake
        status, body = self.request({"language": "Spanish", "lines": ["Eres mi media naranja"]})
        self.assertEqual(status, 200)
        self.assertEqual(body["lines"][0]["en"], "You are my other half")
        self.assertEqual(body["provider"], "gemini")
        self.assertEqual(seen["user"], {"source_language": "Spanish", "lines": ["Eres mi media naranja"]})

    def test_provider_error(self):
        os.environ["GROQ_API_KEY"] = "k"

        def boom(*args):
            raise translate.ProviderError(401, "invalid key")

        translate.call_provider = boom
        status, body = self.request({"lines": ["hola"]})
        self.assertEqual(status, 502)
        self.assertIn("401", body["error"])


if __name__ == "__main__":
    unittest.main()
