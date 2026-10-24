import contextlib
import io
import json
import os
import tempfile
import unittest

from minirag import cli

from test_pipeline import DOC_A, DOC_B, DOC_C


def run(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli.main(argv)
    return code, out.getvalue()


class CLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.corpus = os.path.join(self.tmp.name, "corpus")
        os.makedirs(os.path.join(self.corpus, "sub"))
        for name, text in (("a.md", DOC_A), ("b.txt", DOC_B), (os.path.join("sub", "c.md"), DOC_C)):
            with open(os.path.join(self.corpus, name), "w", encoding="utf-8") as fh:
                fh.write(text)
        with open(os.path.join(self.corpus, "ignored.pdf"), "wb") as fh:
            fh.write(b"%PDF")
        self.index_path = os.path.join(self.tmp.name, "index.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_index_then_ask_then_eval(self):
        code, out = run(["index", self.corpus, "-o", self.index_path, "--chunk-size", "120", "--overlap", "20"])
        self.assertEqual(code, 0)
        self.assertIn("3 documents", out)
        with open(self.index_path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(sorted(c["doc_id"] for c in data["chunks"]), sorted(
            [c["doc_id"] for c in data["chunks"]]))
        self.assertEqual(sorted({c["doc_id"] for c in data["chunks"]}), ["a.md", "b.txt", "sub/c.md"])

        code, out = run(["ask", self.index_path, "What does the KV cache store?", "-k", "3"])
        self.assertEqual(code, 0)
        self.assertIn("Answer", out)
        self.assertIn("keys and values", out)
        self.assertIn("Citations", out)
        self.assertIn("b.txt", out)

        qa_path = os.path.join(self.tmp.name, "qa.json")
        with open(qa_path, "w", encoding="utf-8") as fh:
            json.dump(
                [
                    {"question": "What does the KV cache store?", "gold_doc": "b.txt"},
                    {"question": "When do cats hunt?", "gold_doc": "a.md"},
                ],
                fh,
            )
        code, out = run(["eval", self.index_path, qa_path, "-k", "3"])
        self.assertEqual(code, 0)
        self.assertIn("recall@3", out)
        self.assertIn("1.000", out)
        self.assertIn("MRR", out)

    def test_index_empty_dir_fails_cleanly(self):
        empty = os.path.join(self.tmp.name, "empty")
        os.makedirs(empty)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code, _ = run(["index", empty, "-o", self.index_path])
        self.assertNotEqual(code, 0)
        self.assertIn("no .txt or .md", err.getvalue())

    def test_ask_missing_index_fails_cleanly(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code, _ = run(["ask", os.path.join(self.tmp.name, "nope.json"), "q"])
        self.assertNotEqual(code, 0)
        self.assertIn("nope.json", err.getvalue())

    def test_openai_llm_selected(self):
        run(["index", self.corpus, "-o", self.index_path])
        llm = cli.make_llm("openai", base_url="http://localhost:11434/v1", model="llama3.2", api_key_env="X")
        self.assertEqual(type(llm).__name__, "OpenAICompatibleLLM")
        self.assertEqual(llm.model, "llama3.2")
        self.assertEqual(type(cli.make_llm("extractive")).__name__, "ExtractiveLLM")

    def test_no_command_prints_help(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code, out = run([])
        self.assertNotEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
