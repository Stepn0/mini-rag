import unittest

from minirag.bm25 import BM25Index
from minirag.chunking import Chunk
from minirag.models import Scored
from minirag.rerank import LexicalReranker, Reranker


def chunk(cid, text):
    return Chunk(id=cid, doc_id=cid, text=text, start=0, end=len(text))


class LexicalRerankerTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            chunk("adjacent", "The KV cache stores keys and values for every past token."),
            chunk("scattered", "Keys are stored in one place; the cache lives in a KV layout."),
            chunk("partial", "A cache is a small fast memory."),
            chunk("unrelated", "Bananas are yellow fruit that grow in bunches."),
        ]
        self.index = BM25Index()
        for c in self.chunks:
            self.index.add(c)
        self.reranker = LexicalReranker(self.index)

    def test_ordering_rewards_coverage_and_phrase(self):
        ranked = self.reranker.rerank("What does the KV cache store?", self.chunks)
        ids = [s.chunk_id for s in ranked]
        self.assertEqual(ids[0], "adjacent")
        self.assertEqual(ids[1], "scattered")
        self.assertEqual(ids[-1], "unrelated")
        self.assertTrue(all(isinstance(s, Scored) for s in ranked))
        self.assertTrue(all(a.score >= b.score for a, b in zip(ranked, ranked[1:])))

    def test_components_are_bounded(self):
        parts = self.reranker.explain("KV cache store", self.chunks[0])
        for key in ("bm25", "coverage", "phrase"):
            self.assertGreaterEqual(parts[key], 0.0)
            self.assertLessEqual(parts[key], 1.0)
        self.assertEqual(parts["coverage"], 1.0)
        self.assertEqual(parts["phrase"], 1.0)
        self.assertEqual(self.reranker.explain("KV cache store", self.chunks[-1])["coverage"], 0.0)

    def test_phrase_bonus_breaks_coverage_ties(self):
        shuffled = [chunk("b", "stores cache KV things"), chunk("a", "the KV cache stores")]
        index = BM25Index()
        for c in shuffled:
            index.add(c)
        ranked = LexicalReranker(index).rerank("KV cache stores", shuffled)
        self.assertEqual(ranked[0].chunk_id, "a")

    def test_weights_are_configurable(self):
        only_bm25 = LexicalReranker(self.index, w_bm25=1.0, w_coverage=0.0, w_phrase=0.0)
        ranked = only_bm25.rerank("KV cache store", self.chunks)
        self.assertAlmostEqual(ranked[0].score, 1.0)  # normalized to the max

    def test_empty_inputs(self):
        self.assertEqual(self.reranker.rerank("anything", []), [])
        ranked = self.reranker.rerank("", self.chunks)
        self.assertEqual(len(ranked), len(self.chunks))
        self.assertTrue(all(s.score == 0.0 for s in ranked))

    def test_protocol_accepts_custom_reranker(self):
        class Reverse:
            def rerank(self, query, candidates):
                return [Scored(c.id, float(i)) for i, c in enumerate(candidates)][::-1]

        r: Reranker = Reverse()
        self.assertEqual(r.rerank("q", self.chunks)[0].chunk_id, "unrelated")
        self.assertIsInstance(r, Reranker)


if __name__ == "__main__":
    unittest.main()
