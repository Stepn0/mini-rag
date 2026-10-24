"""The end-to-end RAG pipeline.

    add_document -> chunk -> index (BM25 + TF-IDF)
    ask -> BM25 top-N | TF-IDF top-N -> RRF -> rerank -> prompt with [n] blocks -> LLM
        -> Answer(text, citations, prompt)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .bm25 import BM25Index
from .chunking import Chunk, chunk_text
from .fusion import reciprocal_rank_fusion
from .llm import LLM, ExtractiveLLM
from .models import Scored
from .rerank import LexicalReranker, Reranker
from .tfidf import TFIDFIndex

INDEX_FORMAT = "minirag-index/1"

PROMPT_INSTRUCTIONS = (
    "You are a precise assistant. Answer the question using ONLY the numbered context "
    "passages below. After every claim, cite the passage that supports it with its number "
    "in square brackets, e.g. [2]. If the context does not contain the answer, say so."
)

_CITATION_RE = re.compile(r"\[(\d+)\]")


class NoDocumentsError(ValueError):
    """Raised when asking a question before any document was added."""


@dataclass(frozen=True)
class Citation:
    doc_id: str
    chunk_id: str
    score: float
    snippet: str
    n: int  # the [n] marker used in the prompt/answer


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    prompt: str = ""

    @property
    def cited(self) -> list[Citation]:
        """Citations actually referenced as ``[n]`` in ``text``, in order of first mention."""
        by_n = {c.n: c for c in self.citations}
        seen: list[Citation] = []
        for raw in _CITATION_RE.findall(self.text):
            c = by_n.get(int(raw))
            if c is not None and c not in seen:
                seen.append(c)
        return seen


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    blocks = [f"[{i}] (source: {c.doc_id})\n{c.text.strip()}" for i, c in enumerate(chunks, 1)]
    context = "\n\n".join(blocks) if blocks else "(no context retrieved)"
    return f"{PROMPT_INSTRUCTIONS}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def make_snippet(text: str, width: int = 160) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1].rstrip() + "…"


class RAG:
    def __init__(
        self,
        llm: LLM | None = None,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        strategy: str = "sentence",
        reranker: Reranker | None = None,
        candidates: int = 20,
    ):
        self.llm: LLM = llm if llm is not None else ExtractiveLLM()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.candidates = candidates
        self.chunks: dict[str, Chunk] = {}
        self.doc_ids: list[str] = []
        self.bm25 = BM25Index()
        self.tfidf = TFIDFIndex()
        self.reranker: Reranker = reranker if reranker is not None else LexicalReranker(self.bm25)
        self._pending: list[Chunk] = []

    # ---- building -----------------------------------------------------------------
    def add_document(self, doc_id: str, text: str) -> list[Chunk]:
        if doc_id in self.doc_ids:
            raise ValueError(f"document {doc_id!r} already added")
        chunks = chunk_text(
            text, size=self.chunk_size, overlap=self.chunk_overlap, strategy=self.strategy, doc_id=doc_id
        )
        self.doc_ids.append(doc_id)
        self._pending.extend(chunks)
        return chunks

    def index(self) -> int:
        """Index any documents added since the last call. Returns the number of new chunks."""
        n = len(self._pending)
        for chunk in self._pending:
            self.chunks[chunk.id] = chunk
            self.bm25.add(chunk)
            self.tfidf.add(chunk)
        self._pending.clear()
        return n

    # ---- querying -----------------------------------------------------------------
    def retrieve(self, question: str, k: int = 5) -> list[Scored]:
        """Hybrid retrieval: BM25 and TF-IDF candidates fused with RRF, then reranked."""
        if not self.doc_ids:
            raise NoDocumentsError("no documents indexed; call add_document() before asking")
        if not question.strip():
            raise ValueError("question must not be empty")
        self.index()
        n = max(self.candidates, k)
        lexical = [s.chunk_id for s in self.bm25.search(question, n)]
        vector = [s.chunk_id for s in self.tfidf.search(question, n)]
        fused = reciprocal_rank_fusion([lexical, vector])
        candidates = [self.chunks[s.chunk_id] for s in fused[:n]]
        return self.reranker.rerank(question, candidates)[:k]

    def ask(self, question: str, k: int = 5) -> Answer:
        hits = self.retrieve(question, k)
        chunks = [self.chunks[h.chunk_id] for h in hits]
        prompt = build_prompt(question, chunks)
        text = self.llm.complete(prompt)
        citations = [
            Citation(doc_id=c.doc_id, chunk_id=c.id, score=h.score, snippet=make_snippet(c.text), n=i)
            for i, (h, c) in enumerate(zip(hits, chunks), 1)
        ]
        return Answer(text=text, citations=citations, prompt=prompt)

    # ---- persistence --------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        self.index()
        payload = {
            "format": INDEX_FORMAT,
            "config": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "strategy": self.strategy,
                "candidates": self.candidates,
            },
            "doc_ids": self.doc_ids,
            "chunks": [c.to_dict() for c in self.chunks.values()],
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, llm: LLM | None = None, reranker: Reranker | None = None) -> "RAG":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != INDEX_FORMAT:
            raise ValueError(f"{path}: not a {INDEX_FORMAT} file")
        cfg = data["config"]
        rag = cls(
            llm=llm,
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            strategy=cfg["strategy"],
            reranker=reranker,
            candidates=cfg.get("candidates", 20),
        )
        rag.doc_ids = list(data["doc_ids"])
        rag._pending = [Chunk.from_dict(c) for c in data["chunks"]]
        rag.index()
        return rag
