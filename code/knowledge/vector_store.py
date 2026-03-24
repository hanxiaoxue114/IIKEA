"""
ChromaDB vector store wrapper for II-KEA.

Uses Sentence-BERT (all-MPNet-base-v2) for embeddings, consistent with the paper.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embedding_model: str = "all-MPNet-base-v2",
    ) -> None:
        self._encoder = SentenceTransformer(embedding_model)

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write ──────────────────────────────────────────────────────────────

    def add(
        self,
        texts: List[str],
        metadatas: List[Dict],
        ids: List[str],
    ) -> None:
        embeddings = self._encoder.encode(texts, show_progress_bar=False).tolist()
        self._collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    # ── Query ──────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Returns top-k documents as a list of dicts with keys:
          id, document, metadata, distance
        """
        query_embedding = self._encoder.encode([query_text]).tolist()
        kwargs = dict(
            query_embeddings=query_embedding,
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        docs = []
        for i in range(len(results["ids"][0])):
            docs.append({
                "id":       results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return docs

    def get_metadata(self) -> str:
        """Return a human-readable description of the collection for LLM prompts."""
        count = self._collection.count()
        return (
            f"A medical knowledge database with {count} documents scraped from Wikipedia. "
            "Each document covers a section (Overview, Causes, Signs and Symptoms, "
            "Diagnosis, Treatment, Epidemiology, etc.) of an ICD-9 coded disease. "
            "Documents are indexed by semantic similarity using Sentence-BERT embeddings."
        )

    def __len__(self) -> int:
        return self._collection.count()
