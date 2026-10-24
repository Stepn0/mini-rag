"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, 2009).

RRF(d) = sum over rankings r of 1 / (k + rank_r(d)), rank starting at 1.

It merges ranked lists whose scores live on incompatible scales (BM25 vs. cosine)
without any normalization or tuning; ``k`` (default 60) damps the advantage of
being first in a single list so that consensus across lists wins.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from .models import Scored


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> list[Scored]:
    """Fuse several ranked id lists into one list of :class:`Scored`, best first."""
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [Scored(item, score) for item, score in ranked]
