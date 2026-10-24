import unittest

from minirag.chunking import Chunk
from minirag.tfidf import TFIDFIndex


def chunk(cid, text):
    return Chunk(id=cid, doc_id=cid, text=text, start=0, end=len(text))


class TFIDFTests(unittest.TestCase):
    def setUp(self):
        self.index = TFIDFIndex(tokenizer=str.split)
        self.index.add(chunk("a", "apple banana"))
        self.index.add(chunk("b", "apple banana"))
        self.index.add(chunk("c", "cherry"))
        self.index.add(chunk("d", "apple apple apple durian"))

    def test_identical_document_scores_one(self):
        self.assertAlmostEqual(self.index.similarity("apple banana", "a"), 1.0, places=9)
        self.assertAlmostEqual(self.index.similarity("cherry", "c"), 1.0, places=9)

    def test_orthogonal_document_scores_zero(self):
        self.assertEqual(self.index.similarity("cherry", "a"), 0.0)
        self.assertEqual(self.index.similarity("zebra", "a"), 0.0)

    def test_search_returns_only_positive_matches_sorted(self):
        results = self.index.search("apple banana", k=10)
        ids = [s.chunk_id for s in results]
        self.assertEqual(set(ids), {"a", "b", "d"})
        self.assertEqual(ids[:2], ["a", "b"])
        self.assertTrue(all(x.score >= y.score for x, y in zip(results, results[1:])))
        self.assertTrue(all(0.0 < s.score <= 1.0 + 1e-9 for s in results))

    def test_k_truncates(self):
        self.assertEqual(len(self.index.search("apple", k=1)), 1)

    def test_empty_query(self):
        self.assertEqual(self.index.search("", k=5), [])

    def test_rare_term_weighs_more_than_common(self):
        # "durian" occurs once, "apple" in three docs. Query both; doc d has both,
        # but a doc that only matched the rare term should beat one matching only the common one.
        idx = TFIDFIndex(tokenizer=str.split)
        idx.add(chunk("common_only", "apple"))
        idx.add(chunk("rare_only", "durian"))
        idx.add(chunk("x", "apple"))
        idx.add(chunk("y", "apple"))
        r = {s.chunk_id: s.score for s in idx.search("apple durian", k=10)}
        self.assertGreater(r["rare_only"], r["common_only"])

    def test_incremental_add_recomputes_idf(self):
        idx = TFIDFIndex(tokenizer=str.split)
        idx.add(chunk("a", "apple banana"))
        before = idx.similarity("apple", "a")
        idx.add(chunk("b", "apple"))
        idx.add(chunk("c", "apple"))
        after = idx.similarity("apple", "a")
        self.assertLess(after, before)  # apple got more common, banana now dominates a

    def test_duplicate_chunk_rejected(self):
        with self.assertRaises(ValueError):
            self.index.add(chunk("a", "again"))


if __name__ == "__main__":
    unittest.main()
