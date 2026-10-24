import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from minirag.chunking import Chunk
from minirag.llm import LLM, ExtractiveLLM, LLMError, OpenAICompatibleLLM
from minirag.pipeline import build_prompt


def chunk(cid, doc, text):
    return Chunk(id=cid, doc_id=doc, text=text, start=0, end=len(text))


class ExtractiveLLMTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            chunk("a#0", "a", "Cats are small mammals. They sleep for most of the day."),
            chunk("b#0", "b", "The KV cache stores keys and values of past tokens. It makes decoding O(n) per step."),
            chunk("c#0", "c", "# Heading\nBananas are yellow."),
        ]
        self.prompt = build_prompt("What does the KV cache store?", self.chunks)

    def test_returns_best_sentences_with_citation_markers(self):
        answer = ExtractiveLLM(max_sentences=1).complete(self.prompt)
        self.assertEqual(answer, "The KV cache stores keys and values of past tokens. [2]")

    def test_sentences_without_query_terms_are_never_padded_in(self):
        # Only one sentence in the context shares a term with the question, so even
        # with a budget of 3 sentences the answer must contain just that one.
        answer = ExtractiveLLM(max_sentences=3).complete(self.prompt)
        self.assertEqual(answer, "The KV cache stores keys and values of past tokens. [2]")

    def test_reading_order_across_passages(self):
        chunks = [
            chunk("x#0", "x", "Irrelevant opener. The cache helps a little."),
            chunk("y#0", "y", "The KV cache stores keys. It stores values too."),
        ]
        prompt = build_prompt("What does the KV cache store?", chunks)
        answer = ExtractiveLLM(max_sentences=3).complete(prompt)
        # Best sentence is in passage 2, but output follows passage order [1] then [2].
        self.assertEqual(
            answer,
            "The cache helps a little. [1] The KV cache stores keys. [2] It stores values too. [2]",
        )

    def test_no_overlap_gives_honest_fallback(self):
        prompt = build_prompt("Who won the 1998 World Cup?", self.chunks)
        answer = ExtractiveLLM().complete(prompt)
        self.assertIn("could not find", answer.lower())

    def test_markdown_headings_are_never_quoted(self):
        chunks = [chunk("c#0", "c", "# Bananas\nBananas are yellow.")]
        prompt = build_prompt("Tell me about bananas", chunks)
        answer = ExtractiveLLM(max_sentences=3).complete(prompt)
        self.assertEqual(answer, "Bananas are yellow. [1]")

    def test_duplicate_sentences_from_overlapping_chunks_quoted_once(self):
        chunks = [
            chunk("k#0", "k", "Intro text. The KV cache stores keys and values."),
            chunk("k#1", "k", "The KV cache stores keys and values. The cache costs memory."),
        ]
        prompt = build_prompt("What does the KV cache store?", chunks)
        answer = ExtractiveLLM(max_sentences=3).complete(prompt)
        self.assertEqual(answer, "The KV cache stores keys and values. [1] The cache costs memory. [2]")

    def test_satisfies_protocol(self):
        self.assertIsInstance(ExtractiveLLM(), LLM)

    def test_unstructured_prompt_does_not_crash(self):
        self.assertIsInstance(ExtractiveLLM().complete("hello there"), str)


class _FakeHandler(BaseHTTPRequestHandler):
    requests = []
    status = 200
    body = {"choices": [{"message": {"role": "assistant", "content": "  fake answer [1]  "}}]}

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        _FakeHandler.requests.append(
            {"path": self.path, "json": payload, "auth": self.headers.get("Authorization")}
        )
        data = json.dumps(_FakeHandler.body).encode()
        self.send_response(_FakeHandler.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # silence
        pass


class OpenAICompatibleLLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _FakeHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}/v1/"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        _FakeHandler.requests.clear()
        _FakeHandler.status = 200
        os.environ.pop("MINIRAG_TEST_KEY", None)

    def test_posts_chat_completion_and_returns_content(self):
        os.environ["MINIRAG_TEST_KEY"] = "secret-123"
        llm = OpenAICompatibleLLM(self.base_url, model="tiny-model", api_key_env="MINIRAG_TEST_KEY")
        self.assertEqual(llm.complete("hello"), "fake answer [1]")
        req = _FakeHandler.requests[0]
        self.assertEqual(req["path"], "/v1/chat/completions")
        self.assertEqual(req["json"]["model"], "tiny-model")
        self.assertEqual(req["json"]["messages"][-1], {"role": "user", "content": "hello"})
        self.assertEqual(req["json"]["temperature"], 0.0)
        self.assertEqual(req["auth"], "Bearer secret-123")

    def test_no_api_key_sends_no_auth_header(self):
        llm = OpenAICompatibleLLM(self.base_url, model="m", api_key_env="MINIRAG_TEST_KEY")
        llm.complete("x")
        self.assertIsNone(_FakeHandler.requests[0]["auth"])

    def test_http_error_raises_llm_error(self):
        _FakeHandler.status = 429
        llm = OpenAICompatibleLLM(self.base_url, model="m")
        with self.assertRaises(LLMError) as ctx:
            llm.complete("x")
        self.assertIn("429", str(ctx.exception))

    def test_connection_error_raises_llm_error(self):
        llm = OpenAICompatibleLLM("http://127.0.0.1:9/v1", model="m", timeout=1)
        with self.assertRaises(LLMError):
            llm.complete("x")

    def test_satisfies_protocol(self):
        self.assertIsInstance(OpenAICompatibleLLM(self.base_url, model="m"), LLM)


if __name__ == "__main__":
    unittest.main()
