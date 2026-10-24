# mini-rag

**Retrieval-Augmented Generation from scratch in pure Python: chunking, BM25 + TF-IDF hybrid retrieval, reciprocal rank fusion, reranking, citations, evaluation. Zero dependencies, pluggable LLM.**

[![CI](https://github.com/Stepn0/mini-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Stepn0/mini-rag/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Zero dependencies](https://img.shields.io/badge/dependencies-0-brightgreen)

```
docs ──> chunk ──> ┌─ BM25  ──┐ ──> RRF ──> rerank ──> prompt w/ [n] citations ──> LLM ──> answer + citations
                   └─ TF-IDF ─┘
```

## Why

RAG is how LLMs get grounded in private data: retrieve the passages that matter, put them in the prompt, and make the model cite them. Most people only ever touch it through a framework, so the pipeline stays a black box. `mini-rag` is the whole thing in ~1,100 lines of standard-library Python, each stage a small module with its own unit tests (96 tests, hand-computed BM25 scores included). Read it in an afternoon; swap any stage for a real one.

## Quickstart

```bash
git clone https://github.com/Stepn0/mini-rag && cd mini-rag
python3 -m minirag.cli index data/corpus -o index.json      # or: pip install -e . && minirag index ...
python3 -m minirag.cli ask index.json "What does the KV cache store?"
```

Real output (no model, no network; the default `ExtractiveLLM` quotes the best-matching sentences and cites them):

```
$ python3 -m minirag.cli index data/corpus -o index.json
indexed 10 documents into 54 chunks -> index.json

$ python3 -m minirag.cli ask index.json "What does the KV cache store?"
Answer:
  The KV cache removes this redundancy. [1] Because the key and value projections of
  past tokens never change, they are computed once and stored. [1] PagedAttention,
  popularized by vLLM, stores the cache in fixed-size blocks that can be allocated on
  demand, which eliminates fragmentation and allows prefixes shared between requests to
  be cached once. [2]

Citations:
    #  doc                   score  snippet
  [1]  kv_cache.md           1.656  The KV cache removes this redundancy. Because the ke…
  [2]  kv_cache.md           1.500  Multi-query attention takes this to the extreme with…
  [3]  kv_cache.md           1.336  # The KV Cache Autoregressive decoding generates one…
  [4]  kv_cache.md           0.991  A 32k-token context therefore needs roughly 16 GB ju…
  [5]  vector_databases.md   0.557  # Vector Databases A vector database stores high-dim…
```

```
$ python3 -m minirag.cli eval index.json data/qa.json
questions:   12
recall@5:    1.000
MRR:         1.000
latency:     1.79 ms/question (mean, incl. LLM)
```

From Python:

```python
from minirag import RAG

rag = RAG()                                  # ExtractiveLLM, sentence chunking (400 chars, 80 overlap)
rag.add_document("kv_cache.md", open("data/corpus/kv_cache.md").read())
rag.index()

answer = rag.ask("What does the KV cache store?", k=3)
print(answer.text)                           # "... computed once and stored. [1] ..."
for c in answer.citations:                   # ranked passages that were put in the prompt
    print(c.n, c.doc_id, c.chunk_id, round(c.score, 3), c.snippet[:60])
print(answer.cited)                          # only the citations the answer actually referenced
print(answer.prompt)                         # the exact prompt sent to the LLM
```

## The pipeline, stage by stage

| Stage | Module | What it does |
|---|---|---|
| Chunk | `minirag/chunking.py` | Sentence-aware packing (or fixed-size with word-boundary snapping), configurable overlap. Every chunk is an exact `doc[start:end]` slice, so citations map back to character offsets. |
| Tokenize | `minirag/tokenize.py` | Unicode `\w+` tokenizer, small stopword list, Porter-lite stemmer so `stores / stored / storing` collapse to one term. |
| Lexical retrieval | `minirag/bm25.py` | Okapi BM25 over an inverted index (`k1=1.5`, `b=0.75`), Lucene-style non-negative IDF. |
| Vector retrieval | `minirag/tfidf.py` | TF-IDF sparse vectors, L2-normalized, cosine similarity through posting lists. |
| Fusion | `minirag/fusion.py` | Reciprocal Rank Fusion merges the two ranked lists without score normalization. |
| Rerank | `minirag/rerank.py` | `LexicalReranker`: normalized BM25 + query-term coverage + adjacent-bigram phrase bonus. `Reranker` is a `Protocol`, so a cross-encoder drops in unchanged. |
| Prompt + LLM | `minirag/pipeline.py`, `minirag/llm.py` | Numbered `[1] (source: doc)` context blocks, instruction to cite with `[n]`. `ExtractiveLLM` (offline default) or `OpenAICompatibleLLM` (Ollama, vLLM, OpenAI, ... via `urllib`). |
| Evaluate | `minirag/eval.py` | recall@k, MRR and mean latency over `question / gold_doc` pairs. |
| CLI | `minirag/cli.py` | `minirag index`, `minirag ask`, `minirag eval`; the index persists as JSON. |

### BM25 in one formula

```
score(q, d) = Σ_{t ∈ q}  IDF(t) · tf(t,d) · (k1 + 1) / ( tf(t,d) + k1 · (1 − b + b · |d| / avgdl) )

IDF(t) = ln( (N − n_t + 0.5) / (n_t + 0.5) + 1 )
```

Each query term contributes more when it is rare across the corpus (IDF) and when it appears often in the document, but with diminishing returns: `k1` caps how much repeated occurrences can help, and `b` discounts documents that are longer than average because they mention more words by sheer size. The tests check these scores against the formula to 1e-6 on a three-document toy corpus and verify the saturation and length-penalty properties directly.

## Evaluation

`examples/demo.py` indexes `data/corpus` (10 original ~250-word notes on ML systems topics: transformers, attention, KV cache, quantization, vector databases, HNSW, BM25, RAG evaluation, rate limiting, tokenization), asks three questions, then evaluates on `data/qa.json` (12 question / gold-document pairs). Numbers below are from `python3 examples/demo.py` on this repository, CPython 3.14, one laptop core:

| Metric | Value |
|---|---|
| Documents / chunks | 10 / 54 |
| Questions | 12 |
| recall@5 (gold doc among the top-5 cited docs) | **1.000** |
| MRR (over distinct documents) | **1.000** |
| Mean latency per question, including the extractive LLM | **1.79 ms** |

Honesty notes: the corpus and the questions were written together, the questions use the corpus's own vocabulary, and the corpus is tiny. Perfect retrieval here means the plumbing works, not that lexical retrieval is solved; paraphrased questions ("how do I stop my LLM calls from failing under load?") will miss what a dense embedding model would catch, which is the main item on the roadmap.

## Plugging in a real LLM

Anything with a `complete(prompt: str) -> str` method satisfies the `LLM` protocol. `OpenAICompatibleLLM` talks to any `/chat/completions` endpoint with the standard library only.

```bash
ollama pull llama3.2 && ollama serve          # in another terminal
python3 -m minirag.cli ask index.json "What does the KV cache store?" \
    --llm openai --base-url http://localhost:11434/v1 --model llama3.2
```

```python
from minirag import RAG, OpenAICompatibleLLM

llm = OpenAICompatibleLLM("https://api.openai.com/v1", model="gpt-4o-mini", api_key_env="OPENAI_API_KEY")
rag = RAG.load("index.json", llm=llm)
answer = rag.ask("What does the KV cache store?")
print(answer.text)      # model's answer with [n] markers
print(answer.cited)     # Citation objects for the [n] it actually used
```

The prompt the model receives is always available as `answer.prompt` (or `minirag ask ... --show-prompt`), so citation behaviour can be debugged by reading it.

## Design decisions

- **Hybrid retrieval (BM25 + TF-IDF).** BM25 is the strongest zero-training lexical ranker and nails rare exact tokens (`k1`, `429`, `ef_search`). TF-IDF cosine sees the whole query as a vector and prefers passages whose overall term profile matches, so it ranks differently on the same corpus. Two imperfect, differently-biased rankers fused together beat either alone, and the second slot is exactly where a dense embedding index would go.
- **Reciprocal Rank Fusion.** BM25 scores and cosine similarities live on unrelated scales; RRF only looks at ranks (`1 / (60 + rank)`), needs no normalization or tuning, and rewards consensus. Its known example is unit-tested.
- **A second-stage reranker, even a cheap one.** First-stage retrieval optimizes for recall over the whole corpus; the reranker optimizes for precision over 20 candidates where it can afford per-candidate features (full-coverage, phrase adjacency). The `Reranker` protocol is the seam where a cross-encoder goes.
- **Extractive fallback as the default LLM.** A RAG repo that needs an API key to run its own tests is not testable. `ExtractiveLLM` reads the same prompt a real model would, returns the context sentences that best match the question, and tags each with its `[n]`. The whole pipeline, the CLI and CI run offline and deterministically; swapping in a real model changes one argument.
- **Chunks are exact slices.** `Chunk(start, end)` with `text == doc[start:end]` makes citations verifiable down to the character and is checked by a coverage test (every non-whitespace character of a document lands in at least one chunk).
- **Stdlib only.** `re`, `math`, `json`, `dataclasses`, `urllib`, `http.server` (in tests). No NumPy, so the sparse vectors are plain dicts; that is fine for thousands of chunks and keeps every line readable.
- **Left out on purpose.** Dense embeddings (would need a model), ANN indexes, query rewriting, streaming, and a persistent posting store: the JSON index stores chunks and rebuilds the in-memory indexes on load, which is instant at this scale.

## Development

```bash
make test     # python3 -m unittest discover -s tests -v   (also runs under pytest)
make lint     # python3 -m compileall -q minirag tests examples
make demo     # python3 examples/demo.py
```

CI runs the suite, the demo and a CLI smoke test on Python 3.10, 3.12 and 3.13.

## Roadmap

- [ ] Dense retrieval stage behind the same `search(query, k)` interface (hashed n-gram embeddings first, then an optional sentence-transformers adapter) and three-way RRF.
- [ ] Cross-encoder `Reranker` adapter and a side-by-side eval against `LexicalReranker`.
- [ ] Faithfulness / citation-accuracy metrics in `eval.py` (does each `[n]` sentence appear in passage n?).
- [ ] Query decomposition and HyDE-style query rewriting as optional pre-retrieval steps.
- [ ] A larger, independently written QA set so the numbers above stop being a sanity check and start being a benchmark.

## License

MIT, copyright 2026 Stepn0.
