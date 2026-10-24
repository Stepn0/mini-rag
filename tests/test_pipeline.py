import json
import os
import tempfile
import unittest

from minirag.llm import ExtractiveLLM
from minirag.models import Scored
from minirag.pipeline import RAG, Answer, Citation, NoDocumentsError, build_prompt

DOC_A = (
    "Cats are small carnivorous mammals. They have been kept as pets for thousands of years. "
    "A cat sleeps for most of the day and hunts at night. Kittens are born blind."
)
DOC_B = (
    "The KV cache stores the keys and values of previous tokens during autoregressive decoding. "
    "Because cached keys and values never change, each decoding step only computes the newest token. "
    "This makes each step cost O(n) instead of O(n squared)."
)
DOC_C = (
    "Byte-pair encoding builds a vocabulary by repeatedly merging the most frequent adjacent pair "
    "of symbols. Common words become single tokens while rare words are spelled out from pieces."
)


class RecordingLLM:
    def __init__(self):
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return "Stubbed answer [2] and [1]."


def build_rag(llm=None):
    rag = RAG(llm=llm, chunk_size=120, chunk_overlap=20)
    rag.add_document("A", DOC_A)
    rag.add_document("B", DOC_B)
    rag.add_document("C", DOC_C)
    rag.index()
    return rag


class BuildPromptTests(unittest.TestCase):
    def test_numbered_blocks_and_instructions(self):
        rag = build_rag()
        chunks = list(rag.chunks.values())[:2]
        prompt = build_prompt("What is a cat?", chunks)
        self.assertIn("[1] (source: A)", prompt)
        self.assertIn("[2] (source: ", prompt)
        self.assertIn("Question: What is a cat?", prompt)
        self.assertTrue(prompt.rstrip().endswith("Answer:"))
        self.assertIn("[2]", prompt.split("Context:")[0])  # instructs citing with [n]


class RAGTests(unittest.TestCase):
    def test_answer_in_doc_b_cites_doc_b_first(self):
        rag = build_rag()
        answer = rag.ask("What does the KV cache store?", k=3)
        self.assertIsInstance(answer, Answer)
        self.assertTrue(answer.citations)
        self.assertIsInstance(answer.citations[0], Citation)
        self.assertEqual(answer.citations[0].doc_id, "B")
        self.assertEqual(answer.citations[0].n, 1)
        self.assertIn("keys and values", answer.text)
        self.assertIn("[1]", answer.text)
        self.assertIn(answer.citations[0].chunk_id, rag.chunks)
        self.assertTrue(answer.citations[0].snippet)

    def test_other_doc_wins_for_other_question(self):
        rag = build_rag()
        self.assertEqual(rag.ask("How does byte-pair encoding build a vocabulary?").citations[0].doc_id, "C")
        self.assertEqual(rag.ask("When do cats hunt?").citations[0].doc_id, "A")

    def test_citations_sorted_and_limited_to_k(self):
        rag = build_rag()
        answer = rag.ask("tokens", k=2)
        self.assertLessEqual(len(answer.citations), 2)
        self.assertEqual([c.n for c in answer.citations], list(range(1, len(answer.citations) + 1)))
        scores = [c.score for c in answer.citations]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_cited_maps_markers_back_to_chunks(self):
        llm = RecordingLLM()
        rag = build_rag(llm=llm)
        answer = rag.ask("What does the KV cache store?", k=3)
        self.assertEqual(answer.text, "Stubbed answer [2] and [1].")
        self.assertEqual([c.n for c in answer.cited], [2, 1])
        self.assertEqual(answer.prompt, llm.prompts[0])
        self.assertIn("[1] (source: B)", answer.prompt)

    def test_ask_without_documents_raises_clear_error(self):
        rag = RAG()
        with self.assertRaises(NoDocumentsError) as ctx:
            rag.ask("anything")
        self.assertIn("add_document", str(ctx.exception))
        self.assertIsInstance(ctx.exception, ValueError)

    def test_ask_auto_indexes_when_needed(self):
        rag = RAG(chunk_size=120, chunk_overlap=20)
        rag.add_document("B", DOC_B)
        answer = rag.ask("KV cache")
        self.assertEqual(answer.citations[0].doc_id, "B")

    def test_empty_question_is_rejected(self):
        rag = build_rag()
        with self.assertRaises(ValueError):
            rag.ask("   ")

    def test_duplicate_doc_id_rejected(self):
        rag = RAG()
        rag.add_document("A", DOC_A)
        with self.assertRaises(ValueError):
            rag.add_document("A", DOC_A)

    def test_retrieve_returns_scored(self):
        rag = build_rag()
        hits = rag.retrieve("KV cache", k=2)
        self.assertTrue(hits)
        self.assertIsInstance(hits[0], Scored)
        self.assertTrue(rag.chunks[hits[0].chunk_id].doc_id == "B")

    def test_no_match_returns_empty_citations(self):
        rag = build_rag()
        answer = rag.ask("zebra quokka")
        self.assertEqual(answer.citations, [])
        self.assertIn("could not find", answer.text.lower())

    def test_custom_reranker_is_used(self):
        class Reverse:
            def rerank(self, query, candidates):
                return [Scored(c.id, float(i)) for i, c in enumerate(candidates)][::-1]

        rag = RAG(chunk_size=120, chunk_overlap=20, reranker=Reverse())
        rag.add_document("A", DOC_A)
        rag.add_document("B", DOC_B)
        hits = rag.retrieve("KV cache keys values", k=10)
        self.assertEqual(hits, sorted(hits, key=lambda s: -s.score))

    def test_save_and_load_roundtrip(self):
        rag = build_rag()
        before = rag.ask("What does the KV cache store?")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "index.json")
            rag.save(path)
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(data["format"], "minirag-index/1")
            loaded = RAG.load(path, llm=ExtractiveLLM())
        after = loaded.ask("What does the KV cache store?")
        self.assertEqual(after.text, before.text)
        self.assertEqual([c.chunk_id for c in after.citations], [c.chunk_id for c in before.citations])
        self.assertEqual(loaded.doc_ids, ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
