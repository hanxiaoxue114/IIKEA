"""
Build the domain knowledge vector database for II-KEA.

For each ICD-9 disease, scrapes its Wikipedia page and splits it into
sections (Overview, Signs and Symptoms, Causes, Diagnosis, Prevention,
Treatment, Epidemiology, History, Terminology, Society and Culture).
Each section is embedded with Sentence-BERT and stored in ChromaDB.

Usage:
    python -m knowledge.build_db --disease_list data/processed/disease_list.pkl
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import wikipediaapi
from tqdm import tqdm

from knowledge.vector_store import VectorStore

WIKIPEDIA_SECTIONS = [
    "Overview",
    "Signs and symptoms",
    "Causes",
    "Diagnosis",
    "Prevention",
    "Treatment",
    "Epidemiology",
    "History",
    "Terminology",
    "Society and culture",
]


def icd9_to_name(code: str) -> str:
    """
    Very lightweight ICD-9 code -> human-readable name lookup.
    Falls back to the raw code if not found.
    In production, use a full ICD-9 code mapping CSV.
    """
    # Minimal mapping for demonstration; extend with a full ICD-9 CSV as needed.
    _MAP: Dict[str, str] = {
        "401": "Hypertension",
        "250": "Diabetes mellitus",
        "414": "Coronary artery disease",
        "428": "Heart failure",
        "585": "Chronic kidney disease",
        "486": "Pneumonia",
        "410": "Myocardial infarction",
        "427": "Cardiac arrhythmia",
        "434": "Occlusion of cerebral arteries",
        "496": "Chronic obstructive pulmonary disease",
    }
    prefix = code[:3]
    return _MAP.get(prefix, code)


def fetch_wikipedia_sections(
    disease_name: str,
    wiki: wikipediaapi.Wikipedia,
) -> List[Dict[str, str]]:
    """
    Fetch sections of a Wikipedia page for the given disease.
    Returns a list of {"section": ..., "text": ..., "disease": ...} dicts.
    """
    page = wiki.page(disease_name)
    if not page.exists():
        # Try with disambiguation suffix
        page = wiki.page(f"{disease_name} (medical condition)")
    if not page.exists():
        return []

    docs = []
    # Add the summary as "Overview"
    if page.summary:
        docs.append({
            "section": "Overview",
            "text": page.summary[:2000],
            "disease": disease_name,
        })

    def _walk(sections, target_names):
        for s in sections:
            if any(t.lower() in s.title.lower() for t in target_names):
                text = s.text.strip()
                if text:
                    docs.append({
                        "section": s.title,
                        "text": text[:2000],
                        "disease": disease_name,
                    })
            _walk(s.sections, target_names)

    _walk(page.sections, WIKIPEDIA_SECTIONS)
    return docs


def build_knowledge_db(
    disease_list: List[str],
    vector_store: VectorStore,
    icd9_to_name_fn=icd9_to_name,
    sleep_secs: float = 0.5,
) -> None:
    """
    Scrape Wikipedia for each disease and add documents to the vector store.
    """
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="IIKEA-Research-Agent/1.0",
    )

    added = 0
    for code in tqdm(disease_list, desc="Building knowledge DB"):
        name = icd9_to_name_fn(code)
        docs = fetch_wikipedia_sections(name, wiki)
        if not docs:
            continue

        texts = [d["text"] for d in docs]
        metadatas = [
            {"disease_code": code, "disease_name": name, "section": d["section"]}
            for d in docs
        ]
        ids = [f"{code}_{i}" for i in range(len(docs))]

        vector_store.add(texts=texts, metadatas=metadatas, ids=ids)
        added += len(docs)
        time.sleep(sleep_secs)

    print(f"Added {added} documents for {len(disease_list)} diseases.")


if __name__ == "__main__":
    import argparse
    from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL

    parser = argparse.ArgumentParser()
    parser.add_argument("--disease_list", required=True, help="Path to disease_list.pkl")
    parser.add_argument("--chroma_dir", default=CHROMA_PERSIST_DIR)
    args = parser.parse_args()

    with open(args.disease_list, "rb") as f:
        disease_list = pickle.load(f)

    store = VectorStore(
        persist_dir=args.chroma_dir,
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
    )
    build_knowledge_db(disease_list, store)
    print("Knowledge DB built successfully.")
