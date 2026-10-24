"""Second-stage reranking of a small candidate set.

``LexicalReranker`` is cheap and dependency-free:

    score = w_bm25 * BM25 / max BM25 among candidates
          + w_coverage * (fraction of distinct query terms present in the chunk)
          + w_phrase * (fraction of adjacent query-term bigrams that also appear adjacent)

Anything with a ``rerank(query, candidates) -> list[Scored]`` method satisfies
the :class:`Reranker` protocol, so a cross-encoder can be dropped in unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

from .bm25 import BM25Index
from .chunking import Chunk
from .models import Scored
from .tokenize import tokenize


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[Chunk]) -> list[Scored]:
        """Return every candidate with a new score, best first."""


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


@dataclass
class LexicalReranker:
    bm25: BM25Index
    w_bm25: float = 1.0
    w_coverage: float = 0.5
    w_phrase: float = 0.5
    tokenizer: Callable[[str], list[str]] = field(default=tokenize, repr=False)

    def explain(self, query: str, chunk: Chunk, max_bm25: float | None = None) -> dict[str, float]:
        """Return the individual score components for one candidate (all in [0, 1])."""
        q = self.tokenizer(query)
        qset, qbigrams = set(q), _bigrams(q)
        toks = self.tokenizer(chunk.text)
        raw = self.bm25.score(query, chunk.id)
        denom = max_bm25 if max_bm25 is not None else raw
        return {
            "bm25": raw / denom if denom else 0.0,
            "coverage": len(qset & set(toks)) / len(qset) if qset else 0.0,
            "phrase": len(qbigrams & _bigrams(toks)) / len(qbigrams) if qbigrams else 0.0,
        }

    def rerank(self, query: str, candidates: Sequence[Chunk]) -> list[Scored]:
        if not candidates:
            return []
        max_bm25 = max(self.bm25.score(query, c.id) for c in candidates)
        out = []
        for c in candidates:
            parts = self.explain(query, c, max_bm25=max_bm25)
            total = (
                self.w_bm25 * parts["bm25"]
                + self.w_coverage * parts["coverage"]
                + self.w_phrase * parts["phrase"]
            )
            out.append(Scored(c.id, total))
        return sorted(out, key=lambda s: (-s.score, s.chunk_id))
