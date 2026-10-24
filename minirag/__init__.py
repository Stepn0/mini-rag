"""mini-rag: Retrieval-Augmented Generation from scratch in pure Python.

docs -> chunk -> [BM25 | TF-IDF] -> RRF -> rerank -> prompt with citations -> LLM -> answer
"""

from .bm25 import BM25Index
from .chunking import Chunk, chunk_text
from .eval import evaluate
from .fusion import reciprocal_rank_fusion
from .llm import LLM, ExtractiveLLM, LLMError, OpenAICompatibleLLM
from .models import Scored
from .pipeline import RAG, Answer, Citation, NoDocumentsError, build_prompt
from .rerank import LexicalReranker, Reranker
from .tfidf import TFIDFIndex
from .tokenize import stem, tokenize

__version__ = "0.1.0"

__all__ = [
    "RAG",
    "Answer",
    "Citation",
    "NoDocumentsError",
    "build_prompt",
    "Chunk",
    "chunk_text",
    "BM25Index",
    "TFIDFIndex",
    "Scored",
    "reciprocal_rank_fusion",
    "Reranker",
    "LexicalReranker",
    "LLM",
    "ExtractiveLLM",
    "OpenAICompatibleLLM",
    "LLMError",
    "evaluate",
    "tokenize",
    "stem",
    "__version__",
]
