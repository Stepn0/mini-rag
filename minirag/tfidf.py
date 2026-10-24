"""TF-IDF vectors with cosine similarity, over sparse ``dict`` vectors.

Weights are ``tf * (ln(N / df) + 1)`` and every vector is L2-normalized, so the
cosine similarity is a plain dot product. IDF depends on the corpus size, so
document vectors are (re)built lazily after any ``add``.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Callable

from .chunking import Chunk
from .models import Scored
from .tokenize import tokenize

Tokenizer = Callable[[str], list[str]]
SparseVector = dict[str, float]


class TFIDFIndex:
    def __init__(self, tokenizer: Tokenizer = tokenize):
        self.tokenizer = tokenizer
        self._counts: dict[str, Counter] = {}  # chunk_id -> term counts
        self._df: Counter = Counter()
        self._vectors: dict[str, SparseVector] = {}
        self._postings: dict[str, set[str]] = defaultdict(set)
        self._dirty = False

    def __len__(self) -> int:
        return len(self._counts)

    def add(self, chunk: Chunk) -> None:
        if chunk.id in self._counts:
            raise ValueError(f"chunk {chunk.id!r} already indexed")
        counts = Counter(self.tokenizer(chunk.text))
        self._counts[chunk.id] = counts
        for term in counts:
            self._df[term] += 1
            self._postings[term].add(chunk.id)
        self._dirty = True

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log(len(self._counts) / df) + 1.0 if df else 0.0

    def _vectorize(self, counts: Counter) -> SparseVector:
        vec = {t: c * self._idf(t) for t, c in counts.items() if self._df.get(t)}
        norm = math.sqrt(sum(w * w for w in vec.values()))
        return {t: w / norm for t, w in vec.items()} if norm else {}

    def _ensure_built(self) -> None:
        if self._dirty:
            self._vectors = {cid: self._vectorize(c) for cid, c in self._counts.items()}
            self._dirty = False

    def similarity(self, query: str, chunk_id: str) -> float:
        """Cosine similarity between ``query`` and one chunk (0.0 if unknown)."""
        self._ensure_built()
        doc = self._vectors.get(chunk_id)
        if not doc:
            return 0.0
        q = self._vectorize(Counter(self.tokenizer(query)))
        return sum(w * doc.get(t, 0.0) for t, w in q.items())

    def search(self, query: str, k: int = 10) -> list[Scored]:
        """Top-``k`` chunks by cosine similarity; only chunks sharing a term are returned."""
        self._ensure_built()
        q = self._vectorize(Counter(self.tokenizer(query)))
        scores: dict[str, float] = defaultdict(float)
        for term, qw in q.items():
            for chunk_id in self._postings.get(term, ()):
                scores[chunk_id] += qw * self._vectors[chunk_id].get(term, 0.0)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [Scored(cid, s) for cid, s in ranked[:k] if s > 0.0]
