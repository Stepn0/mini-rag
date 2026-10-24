"""Command-line interface.

    minirag index data/corpus -o index.json
    minirag ask index.json "What does the KV cache store?"
    minirag eval index.json data/qa.json
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from . import __version__
from .eval import evaluate, load_qa
from .llm import LLM, ExtractiveLLM, LLMError, OpenAICompatibleLLM
from .pipeline import RAG, Answer

SUPPORTED_SUFFIXES = (".txt", ".md")


def make_llm(
    kind: str,
    base_url: str = "http://localhost:11434/v1",
    model: str = "llama3.2",
    api_key_env: str = "OPENAI_API_KEY",
) -> LLM:
    if kind == "extractive":
        return ExtractiveLLM()
    if kind == "openai":
        return OpenAICompatibleLLM(base_url=base_url, model=model, api_key_env=api_key_env)
    raise ValueError(f"unknown llm {kind!r}")


def format_answer(answer: Answer, width: int = 88) -> str:
    lines = ["Answer:"]
    lines += textwrap.wrap(answer.text, width=width, initial_indent="  ", subsequent_indent="  ") or ["  (empty)"]
    lines += ["", "Citations:"]
    if not answer.citations:
        lines.append("  (none)")
        return "\n".join(lines)
    doc_w = max(len(c.doc_id) for c in answer.citations)
    doc_w = max(doc_w, len("doc"))
    snippet_w = max(20, width - doc_w - 16)
    lines.append(f"  {'#':>3}  {'doc':<{doc_w}}  {'score':>6}  snippet")
    for c in answer.citations:
        snippet = c.snippet if len(c.snippet) <= snippet_w else c.snippet[: snippet_w - 1].rstrip() + "…"
        lines.append(f"  [{c.n}]  {c.doc_id:<{doc_w}}  {c.score:6.3f}  {snippet}")
    return "\n".join(lines)


def cmd_index(args: argparse.Namespace) -> int:
    root = Path(args.directory)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        print(f"error: no .txt or .md files found under {root}", file=sys.stderr)
        return 1
    rag = RAG(chunk_size=args.chunk_size, chunk_overlap=args.overlap, strategy=args.strategy)
    for path in files:
        rag.add_document(path.relative_to(root).as_posix(), path.read_text(encoding="utf-8"))
    n_chunks = rag.index()
    rag.save(args.output)
    print(f"indexed {len(files)} documents into {n_chunks} chunks -> {args.output}")
    return 0


def _load(args: argparse.Namespace) -> RAG:
    llm = make_llm(args.llm, base_url=args.base_url, model=args.model, api_key_env=args.api_key_env)
    return RAG.load(args.index, llm=llm)


def cmd_ask(args: argparse.Namespace) -> int:
    rag = _load(args)
    answer = rag.ask(args.question, k=args.k)
    print(format_answer(answer))
    if args.show_prompt:
        print("\nPrompt:\n" + textwrap.indent(answer.prompt, "  "))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    rag = _load(args)
    qa = load_qa(args.qa)
    result = evaluate(rag, qa, k=args.k)
    print(f"questions:   {result['n']}")
    print(f"recall@{args.k}:    {result['recall_at_k']:.3f}")
    print(f"MRR:         {result['mrr']:.3f}")
    print(f"latency:     {result['mean_latency_ms']:.2f} ms/question (mean, incl. LLM)")
    if result["misses"]:
        print("misses:")
        for q in result["misses"]:
            print(f"  - {q}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minirag", description="Retrieval-Augmented Generation from scratch.")
    parser.add_argument("--version", action="version", version=f"minirag {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_index = sub.add_parser("index", help="chunk and index a directory of .txt/.md files")
    p_index.add_argument("directory")
    p_index.add_argument("-o", "--output", default="index.json")
    p_index.add_argument("--chunk-size", type=int, default=400)
    p_index.add_argument("--overlap", type=int, default=80)
    p_index.add_argument("--strategy", choices=("sentence", "fixed"), default="sentence")
    p_index.set_defaults(func=cmd_index)

    def add_llm_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--llm", choices=("extractive", "openai"), default="extractive")
        p.add_argument("--base-url", default="http://localhost:11434/v1", help="OpenAI-compatible base URL")
        p.add_argument("--model", default="llama3.2")
        p.add_argument("--api-key-env", default="OPENAI_API_KEY", help="env var holding the API key")
        p.add_argument("-k", type=int, default=5, help="number of passages to retrieve")

    p_ask = sub.add_parser("ask", help="answer a question from an index")
    p_ask.add_argument("index")
    p_ask.add_argument("question")
    p_ask.add_argument("--show-prompt", action="store_true")
    add_llm_flags(p_ask)
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="compute recall@k / MRR on a qa.json file")
    p_eval.add_argument("index")
    p_eval.add_argument("qa")
    add_llm_flags(p_eval)
    p_eval.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help(sys.stderr)
        return 2
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
