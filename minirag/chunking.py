"""Split documents into overlapping chunks.

Two strategies:

* ``"sentence"`` packs whole sentences (split on ``.!?`` + whitespace, or line
  breaks) into chunks of at most ``size`` characters; the overlap is made of the
  trailing sentences of the previous chunk. A single sentence longer than
  ``size`` falls back to fixed splitting.
* ``"fixed"`` cuts every ``size`` characters but snaps both ends to whitespace so
  words are never split; the next chunk starts ``overlap`` characters earlier.

Every chunk's ``text`` is an exact slice ``document[start:end]`` so citations can
be mapped back to byte-accurate positions in the source.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")

STRATEGIES = ("sentence", "fixed")


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str
    start: int
    end: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _skip_ws(text: str, pos: int) -> int:
    n = len(text)
    while pos < n and text[pos].isspace():
        pos += 1
    return pos


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` spans of sentences, trimmed, in document order."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        spans.append((pos, match.start()))
        pos = match.end()
    spans.append((pos, len(text)))
    out = []
    for s, e in spans:
        s, e = _trim(text, s, e)
        if e > s:
            out.append((s, e))
    return out


def _fixed_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    n = len(text)
    spans: list[tuple[int, int]] = []
    start = _skip_ws(text, 0)
    while start < n:
        end = min(start + size, n)
        if end < n and not text[end].isspace():
            # We are in the middle of a word: back up to the last whitespace...
            cut = end
            while cut > start and not text[cut].isspace():
                cut -= 1
            if cut > start:
                end = cut
            else:
                # ...or, for a single word longer than ``size``, extend to its end.
                while end < n and not text[end].isspace():
                    end += 1
        s, e = _trim(text, start, end)
        if e > s:
            spans.append((s, e))
        if end >= n:
            break
        nxt = max(end - overlap, start + 1)
        while nxt < end and not text[nxt - 1].isspace():
            nxt += 1  # never start a chunk mid-word
        nxt = _skip_ws(text, nxt)
        if nxt <= start:
            nxt = _skip_ws(text, end)
        start = nxt
    return spans


def _sentence_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    sents = split_sentences(text)
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(sents):
        j = i
        while j + 1 < len(sents) and sents[j + 1][1] - sents[i][0] <= size:
            j += 1
        s0, e0 = sents[i][0], sents[j][1]
        if e0 - s0 > size:  # a lone oversized sentence
            for a, b in _fixed_spans(text[s0:e0], size, overlap):
                spans.append((s0 + a, s0 + b))
        else:
            spans.append((s0, e0))
        if j + 1 >= len(sents):
            break
        # Overlap: re-include trailing sentences that fit in ``overlap`` chars.
        m = j + 1
        used = 0
        while m - 1 > i and used + (sents[m - 1][1] - sents[m - 1][0]) <= overlap:
            m -= 1
            used += sents[m][1] - sents[m][0]
        i = m
    return spans


def chunk_text(
    text: str,
    size: int = 400,
    overlap: int = 80,
    strategy: str = "sentence",
    doc_id: str = "doc",
) -> list[Chunk]:
    """Split ``text`` into :class:`Chunk` objects with ids ``"{doc_id}#{i}"``."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    if not text.strip():
        return []
    if strategy == "sentence":
        spans = _sentence_spans(text, size, overlap)
    else:
        spans = _fixed_spans(text, size, overlap)
    return [
        Chunk(id=f"{doc_id}#{i}", doc_id=doc_id, text=text[s:e], start=s, end=e)
        for i, (s, e) in enumerate(spans)
    ]
