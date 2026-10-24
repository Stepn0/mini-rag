"""Okapi BM25 over an inverted index.

score(q, d) = sum over terms t in q of
    IDF(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

with IDF(t) = ln((N - n_t + 0.5) / (n_t + 0.5) + 1), the Lucene variant that is
never negative. ``k1`` controls term-frequency saturation and ``b`` controls how
strongly long documents are penalized.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable

from .chunking import Chunk
from .models import Scored
from .tokenize import tokenize

Tokenizer = Callable[[str], list[str]]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75, tokenizer: Tokenizer = tokenize):
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self._postings: dict[str, dict[str, int]] = {}  # term -> {chunk_id: tf}
        self._doc_len: dict[str, int] = {}
        self._total_len = 0

    def __len__(self) -> int:
        return len(self._doc_len)

    @property
    def avgdl(self) -> float:
        return self._total_len / len(self._doc_len) if self._doc_len else 0.0

    def add(self, chunk: Chunk) -> None:
        if chunk.id in self._doc_len:
            raise ValueError(f"chunk {chunk.id!r} already indexed")
        tokens = self.tokenizer(chunk.text)
        self._doc_len[chunk.id] = len(tokens)
        self._total_len += len(tokens)
        for term in tokens:
            posting = self._postings.setdefault(term, {})
            posting[chunk.id] = posting.get(chunk.id, 0) + 1

    def idf(self, term: str) -> float:
        n = len(self._postings.get(term, ()))
        N = len(self._doc_len)
        return math.log((N - n + 0.5) / (n + 0.5) + 1.0)

    def _term_score(self, term: str, chunk_id: str, tf: int, idf: float) -> float:
        norm = 1.0 - self.b + self.b * self._doc_len[chunk_id] / self.avgdl
        return idf * tf * (self.k1 + 1.0) / (tf + self.k1 * norm)

    def score(self, query: str, chunk_id: str) -> float:
        """BM25 score of one chunk for ``query`` (0.0 if unknown or no overlap)."""
        if chunk_id not in self._doc_len:
            return 0.0
        total = 0.0
        for term in self.tokenizer(query):
            tf = self._postings.get(term, {}).get(chunk_id, 0)
            if tf:
                total += self._term_score(term, chunk_id, tf, self.idf(term))
        return total

    def search(self, query: str, k: int = 10) -> list[Scored]:
        """Top-``k`` chunks by BM25 score, highest first. Only matching chunks are returned."""
        scores: dict[str, float] = defaultdict(float)
        for term in self.tokenizer(query):
            posting = self._postings.get(term)
            if not posting:
                continue
            idf = self.idf(term)
            for chunk_id, tf in posting.items():
                scores[chunk_id] += self._term_score(term, chunk_id, tf, idf)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [Scored(chunk_id, score) for chunk_id, score in ranked[:k]]
