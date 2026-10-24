"""Retrieval evaluation: recall@k, MRR and latency over question/gold-document pairs.

Both metrics are computed at the *document* level: the ranked citations are
de-duplicated to the order in which distinct documents first appear, so two
chunks from the same document count as one hit at the rank of the first.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

from .pipeline import RAG

QAPair = dict  # {"question": str, "gold_doc": str}


def load_qa(path: str | Path) -> list[QAPair]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of objects")
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "question" not in item or "gold_doc" not in item:
            raise ValueError(f"{path}: item {i} must have 'question' and 'gold_doc' keys")
    return data


def _distinct_docs(answer) -> list[str]:
    docs: list[str] = []
    for c in answer.citations:
        if c.doc_id not in docs:
            docs.append(c.doc_id)
    return docs


def evaluate(rag: RAG, qa_pairs: Sequence[QAPair], k: int = 5) -> dict:
    hits = 0
    rr_total = 0.0
    latency_total = 0.0
    misses: list[str] = []
    for pair in qa_pairs:
        question, gold = pair["question"], pair["gold_doc"]
        started = time.perf_counter()
        answer = rag.ask(question, k=k)
        latency_total += (time.perf_counter() - started) * 1000.0
        docs = _distinct_docs(answer)
        if gold in docs:
            hits += 1
            rr_total += 1.0 / (docs.index(gold) + 1)
        else:
            misses.append(question)
    n = len(qa_pairs)
    return {
        "n": n,
        "k": k,
        "recall_at_k": hits / n if n else 0.0,
        "mrr": rr_total / n if n else 0.0,
        "mean_latency_ms": latency_total / n if n else 0.0,
        "misses": misses,
    }
