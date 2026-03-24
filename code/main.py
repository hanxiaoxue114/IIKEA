"""
II-KEA main inference pipeline — Algorithm 4 in the paper.

Requires:
  1. Preprocessed matrices:  data/processed/A_T.npy, A_D.npy, etc.
  2. Built ChromaDB:         chroma_db/
  3. Azure OpenAI credentials in environment variables.

Usage:
    python main.py \\
        --diagnoses  path/to/DIAGNOSES_ICD.csv \\
        --admissions path/to/ADMISSIONS.csv \\
        --data_dir   data/processed \\
        --chroma_dir chroma_db \\
        --dataset    mimic3
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

import config
from data.preprocess import (
    load_mimic,
    split_patients,
    get_candidate_diseases,
    load_matrices,
)
from knowledge.vector_store import VectorStore
from agents.knowledge_synthesis_agent import KnowledgeSynthesisAgent
from agents.causal_discovery_agent import CausalDiscoveryAgent
from agents.decision_making_agent import DecisionMakingAgent
from evaluate import evaluate, print_results


# ── Single-patient inference ───────────────────────────────────────────────

def run_patient(
    diagnosis_history: List[str],
    A_T: np.ndarray,
    disease_list: List[str],
    A_D: np.ndarray,
    patient_ids: List[str],
    knowledge_agent: KnowledgeSynthesisAgent,
    causal_agent: CausalDiscoveryAgent,
    decision_agent: DecisionMakingAgent,
    clinician_comment: Optional[str] = None,
    inference_patient_ids: Optional[List[str]] = None,
) -> dict:
    """
    Run full II-KEA inference for one patient.

    Returns a dict with:
      - candidate_diseases
      - summaries
      - causal_graph
      - predicted_codes
      - explanation
    """
    # Step 1 — Candidate disease selection (Eq. 4)
    candidates = get_candidate_diseases(
        diagnosis_history, A_T, disease_list, threshold=config.TRANSITION_THRESHOLD
    )

    # Step 2 — Knowledge Synthesis Agent
    summaries = knowledge_agent.run(diagnosis_history, candidates)

    # Step 3 — Causal Discovery Agent
    causal_graph = causal_agent.run(
        diagnosis_history=diagnosis_history,
        candidate_diseases=candidates,
        summaries=summaries,
        inference_patient_ids=inference_patient_ids,
    )

    # Step 4 — Decision-Making Agent
    predicted_codes, explanation = decision_agent.run(
        diagnosis_history=diagnosis_history,
        candidate_diseases=candidates,
        summaries=summaries,
        causal_graph=causal_graph,
        clinician_comment=clinician_comment,
    )

    return {
        "candidate_diseases": candidates,
        "summaries": summaries,
        "causal_graph": causal_graph,
        "predicted_codes": predicted_codes,
        "explanation": explanation,
    }


# ── Batch evaluation ───────────────────────────────────────────────────────

def evaluate_dataset(
    test_patients,
    train_patient_ids: List[str],
    A_T: np.ndarray,
    A_D: np.ndarray,
    disease_list: List[str],
    patient_ids: List[str],
    knowledge_agent: KnowledgeSynthesisAgent,
    causal_agent: CausalDiscoveryAgent,
    decision_agent: DecisionMakingAgent,
    output_path: Optional[str] = None,
) -> dict:
    """
    Evaluate II-KEA on all test patients and compute w-F1, R@10, R@20.

    For each patient:
      - Input: all visits except the last
      - Target: disease codes in the last visit
    """
    y_true: List[List[str]] = []
    y_pred: List[List[str]] = []
    results_log = []

    for pid, visits in tqdm(test_patients.items(), desc="Evaluating"):
        history_visits = visits[:-1]
        target_visit   = visits[-1]

        # Flatten history into a single set of diagnosis codes
        diagnosis_history = list({code for v in history_visits for code in v})

        result = run_patient(
            diagnosis_history=diagnosis_history,
            A_T=A_T,
            disease_list=disease_list,
            A_D=A_D,
            patient_ids=patient_ids,
            knowledge_agent=knowledge_agent,
            causal_agent=causal_agent,
            decision_agent=decision_agent,
            inference_patient_ids=train_patient_ids,
        )

        y_true.append(target_visit)
        y_pred.append(result["predicted_codes"])
        results_log.append({"patient_id": pid, **result, "ground_truth": target_visit})

    metrics = evaluate(y_true, y_pred, disease_list, k_values=config.RECALL_K_VALUES)

    if output_path:
        with open(output_path, "w") as f:
            json.dump({"metrics": metrics, "patients": results_log}, f, indent=2)
        print(f"Results saved to {output_path}")

    return metrics


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="II-KEA Inference")
    parser.add_argument("--diagnoses",  required=True, help="DIAGNOSES_ICD.csv")
    parser.add_argument("--admissions", required=True, help="ADMISSIONS.csv")
    parser.add_argument("--data_dir",   default="data/processed")
    parser.add_argument("--chroma_dir", default=config.CHROMA_PERSIST_DIR)
    parser.add_argument("--dataset",    choices=["mimic3", "mimic4"], default="mimic3")
    parser.add_argument("--output",     default="results.json")
    parser.add_argument(
        "--single_patient",
        nargs="+",
        metavar="ICD9_CODE",
        help="Run on a single patient given diagnosis history codes (for demo)",
    )
    parser.add_argument("--clinician_comment", default=None)
    args = parser.parse_args()

    # ── Load preprocessed matrices ─────────────────────────────────────────
    data_dir = Path(args.data_dir)
    if data_dir.exists():
        print("Loading preprocessed matrices...")
        A_T, A_D, patient_ids, disease_list = load_matrices(args.data_dir)
    else:
        print("Preprocessed data not found. Running preprocessing...")
        from data.preprocess import build_transition_matrix, build_diagnosis_matrix, save_matrices

        all_patients = load_mimic(args.diagnoses, args.admissions)
        splits = config.MIMIC3_SPLITS if args.dataset == "mimic3" else config.MIMIC4_SPLITS
        train_p, val_p, test_p = split_patients(
            all_patients, splits["train"], splits["val"], splits["test"]
        )

        disease_set: set = set()
        for visits in train_p.values():
            for visit in visits:
                disease_set.update(visit)
        disease_list = sorted(disease_set)

        A_T = build_transition_matrix(train_p, disease_list)
        A_D, patient_ids = build_diagnosis_matrix(train_p, disease_list)
        save_matrices(A_T, A_D, patient_ids, disease_list, args.data_dir)

    # ── Initialise agents ──────────────────────────────────────────────────
    store = VectorStore(
        persist_dir=args.chroma_dir,
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_model=config.EMBEDDING_MODEL,
    )
    knowledge_agent  = KnowledgeSynthesisAgent(vector_store=store)
    causal_agent     = CausalDiscoveryAgent(A_D=A_D, patient_ids=patient_ids, disease_list=disease_list)
    decision_agent   = DecisionMakingAgent()

    # ── Single-patient demo mode ───────────────────────────────────────────
    if args.single_patient:
        print(f"\nRunning single-patient inference for: {args.single_patient}")
        result = run_patient(
            diagnosis_history=args.single_patient,
            A_T=A_T,
            disease_list=disease_list,
            A_D=A_D,
            patient_ids=patient_ids,
            knowledge_agent=knowledge_agent,
            causal_agent=causal_agent,
            decision_agent=decision_agent,
            clinician_comment=args.clinician_comment,
        )
        print("\n── Predicted Diagnoses ─────────────────")
        print(json.dumps(result["predicted_codes"], indent=2))
        print("\n── Explanation ─────────────────────────")
        print(result["explanation"])
        print("\n── Causal Graph ────────────────────────")
        print(json.dumps(result["causal_graph"], indent=2))
        return

    # ── Full dataset evaluation ────────────────────────────────────────────
    all_patients = load_mimic(args.diagnoses, args.admissions)
    splits = config.MIMIC3_SPLITS if args.dataset == "mimic3" else config.MIMIC4_SPLITS
    train_p, _, test_p = split_patients(
        all_patients, splits["train"], splits["val"], splits["test"]
    )

    print(f"\nEvaluating on {len(test_p)} test patients...")
    metrics = evaluate_dataset(
        test_patients=test_p,
        train_patient_ids=list(train_p.keys()),
        A_T=A_T,
        A_D=A_D,
        disease_list=disease_list,
        patient_ids=patient_ids,
        knowledge_agent=knowledge_agent,
        causal_agent=causal_agent,
        decision_agent=decision_agent,
        output_path=args.output,
    )
    print_results(metrics)


if __name__ == "__main__":
    main()
