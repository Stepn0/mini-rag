import math
import unittest

from minirag.bm25 import BM25Index
from minirag.chunking import Chunk
from minirag.models import Scored


def chunk(cid, text):
    return Chunk(id=cid, doc_id=cid, text=text, start=0, end=len(text))


class BM25HandComputedTests(unittest.TestCase):
    """Three-document toy corpus with a whitespace tokenizer so numbers are exact."""

    def setUp(self):
        self.index = BM25Index(k1=1.5, b=0.75, tokenizer=str.split)
        self.docs = {
            "d1": "cat sat mat",
            "d2": "dog sat log",
            "d3": "cat cat dog bird fish",
        }
        for cid, text in self.docs.items():
            self.index.add(chunk(cid, text))

    @staticmethod
    def expected(tf, dl, df, N=3, avgdl=11 / 3, k1=1.5, b=0.75):
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

    def test_matches_formula_single_term(self):
        results = {s.chunk_id: s.score for s in self.index.search("cat", k=10)}
        self.assertEqual(set(results), {"d1", "d3"})
        self.assertAlmostEqual(results["d1"], self.expected(tf=1, dl=3, df=2), places=6)
        self.assertAlmostEqual(results["d3"], self.expected(tf=2, dl=5, df=2), places=6)

    def test_multi_term_query_sums_contributions(self):
        results = {s.chunk_id: s.score for s in self.index.search("cat dog", k=10)}
        self.assertAlmostEqual(
            results["d3"],
            self.expected(tf=2, dl=5, df=2) + self.expected(tf=1, dl=5, df=2),
            places=6,
        )
        self.assertAlmostEqual(results["d2"], self.expected(tf=1, dl=3, df=2), places=6)

    def test_score_method_agrees_with_search(self):
        for s in self.index.search("cat dog", k=10):
            self.assertAlmostEqual(self.index.score("cat dog", s.chunk_id), s.score, places=9)
        self.assertEqual(self.index.score("cat", "d2"), 0.0)
        self.assertEqual(self.index.score("cat", "nonexistent"), 0.0)

    def test_results_sorted_and_truncated(self):
        results = self.index.search("cat dog", k=2)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], Scored)
        self.assertGreaterEqual(results[0].score, results[1].score)
        self.assertEqual(results[0].chunk_id, "d3")

    def test_unknown_term_returns_nothing(self):
        self.assertEqual(self.index.search("zebra", k=5), [])
        self.assertEqual(self.index.search("", k=5), [])

    def test_len_and_avgdl(self):
        self.assertEqual(len(self.index), 3)
        self.assertAlmostEqual(self.index.avgdl, 11 / 3)

    def test_duplicate_chunk_rejected(self):
        with self.assertRaises(ValueError):
            self.index.add(chunk("d1", "again"))


class BM25PropertyTests(unittest.TestCase):
    def test_term_in_all_docs_has_lower_weight(self):
        index = BM25Index(tokenizer=str.split)
        index.add(chunk("a", "common rare"))
        index.add(chunk("b", "common other"))
        index.add(chunk("c", "common thing"))
        self.assertLess(index.idf("common"), index.idf("rare"))
        # Same tf and same length, so the whole score ordering follows idf.
        self.assertLess(index.score("common", "a"), index.score("rare", "a"))

    def test_longer_docs_are_penalized(self):
        index = BM25Index(tokenizer=str.split)
        index.add(chunk("short", "cat sat"))
        index.add(chunk("long", "cat sat on a very long mat with many other words in it"))
        results = {s.chunk_id: s.score for s in index.search("cat", k=2)}
        self.assertGreater(results["short"], results["long"])

    def test_b_zero_disables_length_normalization(self):
        index = BM25Index(b=0.0, tokenizer=str.split)
        index.add(chunk("short", "cat sat"))
        index.add(chunk("long", "cat sat on a very long mat with many other words in it"))
        results = {s.chunk_id: s.score for s in index.search("cat", k=2)}
        self.assertAlmostEqual(results["short"], results["long"])

    def test_tf_saturates(self):
        index = BM25Index(b=0.0, tokenizer=str.split)
        index.add(chunk("one", "cat x x x"))
        index.add(chunk("two", "cat cat x x"))
        index.add(chunk("ten", "cat " * 10))
        r = {s.chunk_id: s.score for s in index.search("cat", k=3)}
        # Marginal gain per extra occurrence must shrink (diminishing returns).
        self.assertGreater(r["two"] - r["one"], (r["ten"] - r["two"]) / 8)
        self.assertLess(r["ten"], r["one"] * (1.5 + 1))  # bounded by k1 + 1

    def test_default_tokenizer_applies_stemming(self):
        index = BM25Index()
        index.add(chunk("a", "The cache stores keys."))
        self.assertEqual([s.chunk_id for s in index.search("stored caches")], ["a"])


if __name__ == "__main__":
    unittest.main()
