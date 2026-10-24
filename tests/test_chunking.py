import unittest

from minirag.chunking import Chunk, chunk_text, split_sentences

PROSE = " ".join(
    f"Sentence number {i} talks about topic {i} and nothing else." for i in range(60)
)
MARKDOWN = "# Title\n\nFirst paragraph here. It has two sentences.\n\nSecond paragraph!\nThird line?"


def covered_positions(chunks):
    covered = set()
    for c in chunks:
        covered.update(range(c.start, c.end))
    return covered


class SplitSentencesTests(unittest.TestCase):
    def test_spans_are_trimmed_and_in_order(self):
        spans = split_sentences(MARKDOWN)
        texts = [MARKDOWN[s:e] for s, e in spans]
        self.assertEqual(
            texts,
            ["# Title", "First paragraph here.", "It has two sentences.", "Second paragraph!", "Third line?"],
        )

    def test_empty(self):
        self.assertEqual(split_sentences("   \n "), [])


class ChunkCommonTests(unittest.TestCase):
    def test_returns_chunk_dataclasses_with_ids(self):
        chunks = chunk_text(PROSE, size=200, overlap=40, doc_id="d")
        self.assertTrue(chunks)
        for i, c in enumerate(chunks):
            self.assertIsInstance(c, Chunk)
            self.assertEqual(c.id, f"d#{i}")
            self.assertEqual(c.doc_id, "d")
            self.assertEqual(c.text, PROSE[c.start : c.end])
            self.assertLess(c.start, c.end)

    def test_empty_text_gives_no_chunks(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n\t"), [])

    def test_short_text_is_single_chunk(self):
        for strategy in ("sentence", "fixed"):
            chunks = chunk_text("Just one short sentence.", size=400, strategy=strategy)
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0].text, "Just one short sentence.")

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            chunk_text(PROSE, size=0)
        with self.assertRaises(ValueError):
            chunk_text(PROSE, size=100, overlap=100)
        with self.assertRaises(ValueError):
            chunk_text(PROSE, strategy="magic")

    def test_round_trip_dict(self):
        c = chunk_text(PROSE, size=200, overlap=0)[0]
        self.assertEqual(Chunk.from_dict(c.to_dict()), c)


class FixedChunkingTests(unittest.TestCase):
    def test_sizes_respected(self):
        for size in (50, 120, 400):
            chunks = chunk_text(PROSE, size=size, overlap=10, strategy="fixed")
            self.assertTrue(all(len(c.text) <= size for c in chunks), size)

    def test_never_splits_words(self):
        chunks = chunk_text(PROSE, size=97, overlap=20, strategy="fixed")
        for c in chunks:
            self.assertTrue(c.start == 0 or PROSE[c.start - 1].isspace(), c)
            self.assertTrue(c.end == len(PROSE) or PROSE[c.end].isspace(), c)
            self.assertFalse(c.text[0].isspace())
            self.assertFalse(c.text[-1].isspace())

    def test_overlap_preserved(self):
        chunks = chunk_text(PROSE, size=120, overlap=40, strategy="fixed")
        self.assertGreater(len(chunks), 3)
        for a, b in zip(chunks, chunks[1:]):
            self.assertLess(b.start, a.end, "consecutive chunks should overlap")
            self.assertGreaterEqual(b.start, a.end - 40)
            self.assertGreater(b.start, a.start, "chunks must make progress")

    def test_zero_overlap_is_disjoint(self):
        chunks = chunk_text(PROSE, size=120, overlap=0, strategy="fixed")
        for a, b in zip(chunks, chunks[1:]):
            self.assertGreaterEqual(b.start, a.end)

    def test_full_coverage(self):
        chunks = chunk_text(PROSE, size=90, overlap=30, strategy="fixed")
        covered = covered_positions(chunks)
        for i, ch in enumerate(PROSE):
            if not ch.isspace():
                self.assertIn(i, covered, f"char {i} ({ch!r}) not covered")

    def test_single_giant_word_is_not_lost(self):
        text = "a " + "x" * 500 + " b"
        chunks = chunk_text(text, size=100, overlap=10, strategy="fixed")
        covered = covered_positions(chunks)
        self.assertTrue(all(i in covered for i, ch in enumerate(text) if not ch.isspace()))


class SentenceChunkingTests(unittest.TestCase):
    def test_chunks_end_at_sentence_boundaries(self):
        chunks = chunk_text(PROSE, size=300, overlap=60, strategy="sentence")
        self.assertGreater(len(chunks), 5)
        for c in chunks:
            self.assertTrue(c.text.endswith("."), c.text[-30:])
            self.assertTrue(c.text.startswith("Sentence number"), c.text[:30])
            self.assertLessEqual(len(c.text), 300)

    def test_overlap_repeats_trailing_sentences(self):
        chunks = chunk_text(PROSE, size=300, overlap=60, strategy="sentence")
        for a, b in zip(chunks, chunks[1:]):
            self.assertLess(b.start, a.end)
            self.assertGreater(b.start, a.start)
            # The overlapping region is made of whole sentences.
            shared = PROSE[b.start : a.end]
            self.assertTrue(shared.endswith("."))
            self.assertTrue(shared.startswith("Sentence number"))

    def test_full_coverage(self):
        chunks = chunk_text(PROSE, size=250, overlap=50, strategy="sentence")
        covered = covered_positions(chunks)
        for i, ch in enumerate(PROSE):
            if not ch.isspace():
                self.assertIn(i, covered)

    def test_markdown_headings_and_newlines(self):
        chunks = chunk_text(MARKDOWN, size=40, overlap=0, strategy="sentence")
        covered = covered_positions(chunks)
        self.assertTrue(all(i in covered for i, ch in enumerate(MARKDOWN) if not ch.isspace()))
        self.assertEqual(chunks[0].text.split("\n")[0], "# Title")

    def test_oversized_sentence_falls_back_to_fixed_split(self):
        text = "word " * 300  # 1500 chars, no sentence punctuation
        chunks = chunk_text(text, size=100, overlap=20, strategy="sentence")
        self.assertGreater(len(chunks), 10)
        self.assertTrue(all(len(c.text) <= 100 for c in chunks))
        covered = covered_positions(chunks)
        self.assertTrue(all(i in covered for i, ch in enumerate(text) if not ch.isspace()))


if __name__ == "__main__":
    unittest.main()
