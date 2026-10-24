"""Pluggable LLM back-ends.

* :class:`ExtractiveLLM` needs no network or model: it reads the numbered context
  blocks in the prompt and returns the sentences sharing the most terms with the
  question, each tagged with its ``[n]`` citation. The whole pipeline therefore
  works (and is testable) offline.
* :class:`OpenAICompatibleLLM` POSTs to any ``/chat/completions`` endpoint
  (OpenAI, Ollama, vLLM, LM Studio, ...) with ``urllib`` only.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable

from .chunking import split_sentences
from .tokenize import tokenize

NO_ANSWER = "I could not find an answer in the provided context."

_BLOCK_RE = re.compile(r"^\[(\d+)\] \(source: [^\n]*\)\n", re.MULTILINE)
_HEADING_RE = re.compile(r"^\s*#{1,6}\s")


@runtime_checkable
class LLM(Protocol):
    def complete(self, prompt: str) -> str:
        """Return the model's completion for ``prompt``."""


class LLMError(RuntimeError):
    """Raised when a remote LLM call fails."""


class ExtractiveLLM:
    """Zero-dependency fallback: answer by quoting the most relevant context sentences."""

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    @staticmethod
    def _parse(prompt: str) -> tuple[str, list[tuple[int, str]]]:
        """Split a ``build_prompt`` output into (question, [(n, passage_text), ...])."""
        head, sep, _ = prompt.rpartition("\nQuestion: ")
        if not sep:
            return prompt, []
        question = prompt[len(head) + len(sep):].split("\n", 1)[0]
        matches = list(_BLOCK_RE.finditer(head))
        blocks = []
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(head)
            blocks.append((int(m.group(1)), head[m.end():end].strip()))
        return question, blocks

    def complete(self, prompt: str) -> str:
        question, blocks = self._parse(prompt)
        qterms = set(tokenize(question))
        if not qterms or not blocks:
            return NO_ANSWER
        scored = []
        seen: set[str] = set()
        for rank, (n, passage) in enumerate(blocks):
            for pos, (s, e) in enumerate(split_sentences(passage)):
                raw = passage[s:e]
                if _HEADING_RE.match(raw):
                    continue  # markdown headings are labels, not answers
                sentence = " ".join(raw.split())
                if sentence in seen:
                    continue  # overlapping chunks repeat sentences; quote each once
                seen.add(sentence)
                overlap = len(qterms & set(tokenize(sentence)))
                if overlap:
                    scored.append((-overlap, rank, pos, f"{sentence} [{n}]"))
        if not scored:
            return NO_ANSWER
        # Pick the sentences with the most query-term overlap (ties: earlier passage,
        # earlier position), then emit them in reading order for a coherent answer.
        best = sorted(scored)[: self.max_sentences]
        best.sort(key=lambda item: (item[1], item[2]))
        return " ".join(item[3] for item in best)


class OpenAICompatibleLLM:
    """Chat-completions client for OpenAI-compatible servers (Ollama, vLLM, OpenAI, ...)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 60.0,
        temperature: float = 0.0,
        system_prompt: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.temperature = temperature
        self.system_prompt = system_prompt

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def complete(self, prompt: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        body = json.dumps(
            {"model": self.model, "messages": messages, "temperature": self.temperature}
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise LLMError(f"{self.endpoint} returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise LLMError(f"request to {self.endpoint} failed: {exc}") from exc
        try:
            return payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMError(f"unexpected response shape from {self.endpoint}: {payload!r}") from exc
