"""End-to-end demo: index data/corpus, ask three questions, evaluate on data/qa.json.

Run from the repository root:  python3 examples/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from minirag import RAG, evaluate  # noqa: E402
from minirag.cli import format_answer  # noqa: E402
from minirag.eval import load_qa  # noqa: E402

QUESTIONS = [
    "What does the KV cache store and why does it reduce decoding cost?",
    "What do the k1 and b parameters control in BM25?",
    "How should a client handle an HTTP 429 response from an LLM API?",
]


def main() -> None:
    corpus = ROOT / "data" / "corpus"
    rag = RAG()  # ExtractiveLLM by default: no network, no model download
    for path in sorted(corpus.glob("*.md")):
        rag.add_document(path.name, path.read_text(encoding="utf-8"))
    n_chunks = rag.index()
    print(f"Indexed {len(rag.doc_ids)} documents into {n_chunks} chunks.\n")

    for question in QUESTIONS:
        print("=" * 88)
        print(f"Q: {question}\n")
        print(format_answer(rag.ask(question, k=3)))
        print()

    print("=" * 88)
    qa = load_qa(ROOT / "data" / "qa.json")
    result = evaluate(rag, qa, k=5)
    print(f"Evaluation on {result['n']} questions (k={result['k']}):")
    print(f"  recall@5 = {result['recall_at_k']:.3f}")
    print(f"  MRR      = {result['mrr']:.3f}")
    print(f"  latency  = {result['mean_latency_ms']:.2f} ms/question (mean, incl. extractive LLM)")
    if result["misses"]:
        print("  misses:")
        for q in result["misses"]:
            print(f"    - {q}")


if __name__ == "__main__":
    main()
