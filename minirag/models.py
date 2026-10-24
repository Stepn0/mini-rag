"""Small shared value types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scored:
    """A chunk id paired with a relevance score. Higher is better."""

    chunk_id: str
    score: float
