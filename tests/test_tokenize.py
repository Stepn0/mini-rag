import unittest

from minirag.tokenize import STOPWORDS, stem, tokenize


class StemTests(unittest.TestCase):
    def test_suffix_rules(self):
        cases = {
            "running": "run",        # -ing, undouble consonant
            "embeddings": "embed",   # -s then -ing then undouble
            "indexed": "index",      # -ed
            "fitted": "fit",         # -ed, undouble
            "stored": "store",       # -ed, restore trailing e (cvc rule)
            "queries": "query",      # -ies -> y
            "boxes": "box",          # -xes -> x
            "classes": "class",      # -sses -> ss
            "glass": "glass",        # -ss untouched
            "models": "model",       # -s
            "quickly": "quick",      # -ly
            "attention": "attention",  # no rule applies
        }
        for word, expected in cases.items():
            with self.subTest(word=word):
                self.assertEqual(stem(word), expected)

    def test_inflections_collapse_to_same_stem(self):
        for group in (
            ("store", "stores", "stored", "storing"),
            ("query", "queries"),
            ("cache", "caches", "cached", "caching"),
            ("quantize", "quantized", "quantizing"),
            ("index", "indexes", "indexed", "indexing"),
        ):
            stems = {stem(w) for w in group}
            self.assertEqual(len(stems), 1, f"{group} -> {stems}")

    def test_short_words_untouched(self):
        for w in ("is", "as", "us", "ab"):
            self.assertEqual(stem(w), w)

    def test_no_vowel_stem_is_not_stripped(self):
        # Porter's (*v*) condition: "thing" must not become "th".
        self.assertEqual(stem("thing"), "thing")
        self.assertEqual(stem("string"), "string")


class TokenizeTests(unittest.TestCase):
    def test_lowercase_stopwords_and_stems(self):
        self.assertEqual(tokenize("The Transformers are Running!"), ["transformer", "run"])

    def test_unicode_words_are_kept(self):
        self.assertEqual(tokenize("Zürich café"), ["zürich", "café"])

    def test_punctuation_and_digits(self):
        self.assertEqual(tokenize("k1=1.5, b=0.75"), ["k1", "1", "5", "b", "0", "75"])

    def test_stopwords_are_lowercase_and_removed(self):
        self.assertTrue(all(w == w.lower() for w in STOPWORDS))
        self.assertIn("the", STOPWORDS)
        self.assertEqual(tokenize("the of and"), [])

    def test_can_disable_stemming_and_stopwords(self):
        self.assertEqual(
            tokenize("The caches", stopwords=False, stemming=False), ["the", "caches"]
        )

    def test_empty(self):
        self.assertEqual(tokenize(""), [])


if __name__ == "__main__":
    unittest.main()
