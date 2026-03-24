"""
Knowledge Synthesis Agent  (A_knowledge)  — Algorithm 1 in the paper.

Two-step process:
  1. Generate a search query from the patient's diagnosis history and
     candidate diseases, using the database metadata as context.
  2. For each retrieved document, perform "reasoning-in-documents" to
     produce a concise, relevant summary.
"""

from __future__ import annotations

from typing import List

from openai import AzureOpenAI

from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_DEPLOYMENT_NAME,
    LLM_TEMPERATURE,
    TOP_K_DOCS,
)
from knowledge.vector_store import VectorStore


class KnowledgeSynthesisAgent:
    """
    A_knowledge: retrieves and summarises external medical knowledge relevant
    to a patient's diagnosis history and candidate diseases.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._store = vector_store
        self._client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        diagnosis_history: List[str],
        candidate_diseases: List[str],
        k: int = TOP_K_DOCS,
    ) -> List[str]:
        """
        Returns a list of concise document summaries (Gamma_p^summary).
        """
        query = self._generate_search_query(diagnosis_history, candidate_diseases)
        retrieved_docs = self._store.query(query_text=query, k=k)

        summaries = []
        for doc in retrieved_docs:
            summary = self._reason_in_document(
                document=doc["document"],
                diagnosis_history=diagnosis_history,
                candidate_diseases=candidate_diseases,
            )
            summaries.append(summary)

        return summaries

    # ── Step 1: Generate search query ──────────────────────────────────────

    def _generate_search_query(
        self,
        diagnosis_history: List[str],
        candidate_diseases: List[str],
    ) -> str:
        meta = self._store.get_metadata()
        prompt = (
            "Generate a search query to retrieve the most relevant information "
            "from the knowledge database using the following:\n\n"
            f"Diagnosis history: {', '.join(diagnosis_history)}\n"
            f"Candidate diseases: {', '.join(candidate_diseases)}\n\n"
            "The generated search query should take into account the "
            "characteristics of the knowledge database, as described by the "
            f"provided Meta-data:\n{meta}\n\n"
            "Output only the search query text, nothing else."
        )
        return self._call_llm(prompt)

    # ── Step 2: Reason-in-document ─────────────────────────────────────────

    def _reason_in_document(
        self,
        document: str,
        diagnosis_history: List[str],
        candidate_diseases: List[str],
    ) -> str:
        prompt = (
            f"Summarize the following document.\n\n"
            f"Document:\n{document}\n\n"
            "The output summary should satisfy the following requirements:\n"
            "Relevance: Include only information related to the patient's "
            f"Diagnosis history ({', '.join(diagnosis_history)}) and "
            f"Candidate diseases ({', '.join(candidate_diseases)}).\n"
            "Conciseness: Remove redundant and unnecessary details while "
            "maintaining key insights.\n"
            "Clarity: Ensure the summary is well-structured and easy to understand."
        )
        return self._call_llm(prompt)

    # ── LLM helper ─────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
