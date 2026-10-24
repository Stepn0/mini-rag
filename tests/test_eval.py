import unittest

from minirag.eval import evaluate, load_qa
from minirag.pipeline import RAG

from test_pipeline import DOC_A, DOC_B, DOC_C


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.rag = RAG(chunk_size=120, chunk_overlap=20)
        self.rag.add_document("A", DOC_A)
        self.rag.add_document("B", DOC_B)
        self.rag.add_document("C", DOC_C)
        self.rag.index()

    def test_perfect_set(self):
        qa = [
            {"question": "What does the KV cache store?", "gold_doc": "B"},
            {"question": "How does byte-pair encoding merge symbols?", "gold_doc": "C"},
            {"question": "When do cats hunt?", "gold_doc": "A"},
        ]
        result = evaluate(self.rag, qa, k=3)
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["k"], 3)
        self.assertEqual(result["recall_at_k"], 1.0)
        self.assertEqual(result["mrr"], 1.0)
        self.assertGreater(result["mean_latency_ms"], 0.0)
        self.assertEqual(result["misses"], [])

    def test_partial_credit(self):
        qa = [
            {"question": "What does the KV cache store?", "gold_doc": "B"},
            {"question": "What does the KV cache store?", "gold_doc": "ZZZ"},  # impossible
        ]
        result = evaluate(self.rag, qa, k=3)
        self.assertEqual(result["recall_at_k"], 0.5)
        self.assertEqual(result["mrr"], 0.5)
        self.assertEqual(result["misses"], ["What does the KV cache store?"])

    def test_mrr_uses_rank_of_first_gold_document(self):
        # Gold is A but the question is about the KV cache: B ranks first, so A can be at best rank 2.
        qa = [{"question": "What does the KV cache store?", "gold_doc": "A"}]
        result = evaluate(self.rag, qa, k=3)
        if result["recall_at_k"] == 1.0:
            self.assertLess(result["mrr"], 1.0)
            self.assertGreaterEqual(result["mrr"], 1 / 3)
        else:
            self.assertEqual(result["mrr"], 0.0)

    def test_empty_set(self):
        result = evaluate(self.rag, [], k=3)
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["recall_at_k"], 0.0)

    def test_load_qa_validates(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "qa.json")
            with open(good, "w", encoding="utf-8") as fh:
                json.dump([{"question": "q", "gold_doc": "A"}], fh)
            self.assertEqual(load_qa(good), [{"question": "q", "gold_doc": "A"}])
            bad = os.path.join(tmp, "bad.json")
            with open(bad, "w", encoding="utf-8") as fh:
                json.dump([{"q": "oops"}], fh)
            with self.assertRaises(ValueError):
                load_qa(bad)


if __name__ == "__main__":
    unittest.main()
