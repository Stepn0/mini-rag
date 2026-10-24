import unittest

from minirag.fusion import reciprocal_rank_fusion


class RRFTests(unittest.TestCase):
    def test_known_example(self):
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]], k=60)
        scores = {s.chunk_id: s.score for s in fused}
        self.assertAlmostEqual(scores["a"], 1 / 61 + 1 / 63, places=12)
        self.assertAlmostEqual(scores["b"], 1 / 61 + 1 / 62, places=12)
        self.assertAlmostEqual(scores["c"], 1 / 62 + 1 / 63, places=12)
        self.assertEqual([s.chunk_id for s in fused], ["b", "a", "c"])

    def test_item_in_one_ranking_only(self):
        fused = reciprocal_rank_fusion([["a", "b"], ["c"]], k=1)
        scores = {s.chunk_id: s.score for s in fused}
        self.assertAlmostEqual(scores["a"], 1 / 2)
        self.assertAlmostEqual(scores["c"], 1 / 2)
        self.assertAlmostEqual(scores["b"], 1 / 3)

    def test_consensus_beats_single_top_rank(self):
        # "x" is #1 in one list and absent from the other; "y" is #2 in both.
        fused = reciprocal_rank_fusion([["x", "y", "z"], ["w", "y", "z"]])
        self.assertEqual(fused[0].chunk_id, "y")

    def test_ties_broken_deterministically(self):
        fused = reciprocal_rank_fusion([["b", "a"], ["a", "b"]])
        self.assertEqual([s.chunk_id for s in fused], ["a", "b"])

    def test_empty(self):
        self.assertEqual(reciprocal_rank_fusion([]), [])
        self.assertEqual(reciprocal_rank_fusion([[], []]), [])

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([["a"]], k=-1)


if __name__ == "__main__":
    unittest.main()
