"""
Causal Discovery Agent  (A_causal)  — Algorithm 2 in the paper.

Iterative causal discovery loop:
  1. Hypothesis generation  — LLM produces initial DAG (JSON)
  2. Model fitting          — compute log-likelihood against A_D (Eq. 5)
  3. Post-processing        — update memory M_t
  4. Hypothesis amendment   — LLM refines DAG given memory
  Repeat until stopping criterion or max iterations.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import AzureOpenAI

from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_DEPLOYMENT_NAME,
    LLM_TEMPERATURE,
    MAX_CAUSAL_ITERATIONS,
    CAUSAL_STOP_THRESHOLD,
)


# A DAG is represented as {disease: [parent_diseases]}
DAG = Dict[str, List[str]]


class CausalDiscoveryAgent:
    """
    A_causal: uncovers causal relationships among diseases using iterative
    LLM-guided causal discovery grounded in observational data.
    """

    def __init__(self, A_D: np.ndarray, patient_ids: List[str], disease_list: List[str]) -> None:
        self._A_D = A_D                          # shape (n_patients, n_diseases)
        self._pid2row = {p: i for i, p in enumerate(patient_ids)}
        self._disease_list = disease_list
        self._d2i = {d: i for i, d in enumerate(disease_list)}
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
        summaries: List[str],
        inference_patient_ids: Optional[List[str]] = None,
    ) -> DAG:
        """
        Runs the iterative causal discovery loop (Algorithm 2).

        Args:
            diagnosis_history:     D_p — diseases the patient has had
            candidate_diseases:    C_p — candidate future diseases
            summaries:             Gamma_p^summary from KnowledgeSynthesisAgent
            inference_patient_ids: subset of patients used for fitting (P_ir)

        Returns:
            Final causal graph G^s as a dict {node: [parent_nodes]}
        """
        all_diseases = list(set(diagnosis_history) | set(candidate_diseases))
        summary_text = "\n\n".join(summaries)

        # Step 1 — Hypothesis generation
        dag = self._hypothesis_generation(all_diseases, summary_text)
        memory: List[Tuple[DAG, float]] = []

        for t in range(MAX_CAUSAL_ITERATIONS):
            # Step 2 — Model fitting
            score = self._log_likelihood(dag, inference_patient_ids)

            # Step 3 — Post-processing: update memory
            if memory:
                prev_dag, prev_score = memory[-1]
                memory.append((dag, score))
            else:
                memory.append((dag, score))

            # Step 4 — Hypothesis amendment
            new_dag = self._hypothesis_amendment(memory)

            # Stopping criterion: graph unchanged
            if self._dags_equal(dag, new_dag):
                dag = new_dag
                break

            dag = new_dag

        return dag

    # ── Step 1: Hypothesis generation ─────────────────────────────────────

    def _hypothesis_generation(self, disease_names: List[str], summary: str) -> DAG:
        prompt = (
            "Generate a Directed Acyclic Graph (DAG) to represent the causal "
            "relationships between the given set of disease names. "
            "Use the provided summary, along with contextual knowledge and "
            "reasoning, to infer causality. The output should be in JSON format.\n\n"
            f"Disease names: {json.dumps(disease_names)}\n\n"
            f"Summary:\n{summary}\n\n"
            "Output a JSON object where each key is a disease name and its value "
            "is a list of diseases that are CAUSES of (i.e., parent nodes of) that disease. "
            "Example: {\"Heart failure\": [\"Hypertension\", \"Diabetes\"], \"Hypertension\": []}"
        )
        return self._parse_dag(self._call_llm(prompt), disease_names)

    # ── Step 2: Model fitting  (Eq. 5) ────────────────────────────────────

    def _log_likelihood(
        self,
        dag: DAG,
        patient_ids: Optional[List[str]] = None,
    ) -> float:
        """
        l_t = sum_{p in P_ir} sum_{a in D}
                  log P(X_p^a | {X_p^b : b in Pa(a)})

        P(X^a | parents) is estimated empirically from A_D.
        """
        if patient_ids is not None:
            rows = [self._pid2row[p] for p in patient_ids if p in self._pid2row]
            A = self._A_D[rows]
        else:
            A = self._A_D

        n_patients = A.shape[0]
        if n_patients == 0:
            return 0.0

        ll = 0.0
        for disease, parents in dag.items():
            if disease not in self._d2i:
                continue
            a_idx = self._d2i[disease]
            parent_idxs = [self._d2i[p] for p in parents if p in self._d2i]

            x_a = A[:, a_idx]  # shape (n_patients,)

            if not parent_idxs:
                # Marginal probability
                p_a = x_a.mean()
                p_a = np.clip(p_a, 1e-9, 1 - 1e-9)
                ll += np.sum(
                    x_a * math.log(p_a) + (1 - x_a) * math.log(1 - p_a)
                )
            else:
                # Conditional probability per parent configuration
                parent_matrix = A[:, parent_idxs]  # (n_patients, n_parents)
                configs = defaultdict(lambda: [0, 0])  # config -> [count_a1, count_total]
                for i in range(n_patients):
                    key = tuple(parent_matrix[i].tolist())
                    configs[key][1] += 1
                    configs[key][0] += int(x_a[i])

                for i in range(n_patients):
                    key = tuple(parent_matrix[i].tolist())
                    cnt_a1, cnt_total = configs[key]
                    p_cond = cnt_a1 / cnt_total if cnt_total > 0 else 0.5
                    p_cond = np.clip(p_cond, 1e-9, 1 - 1e-9)
                    ll += int(x_a[i]) * math.log(p_cond) + (1 - int(x_a[i])) * math.log(1 - p_cond)

        return ll

    # ── Step 4: Hypothesis amendment ──────────────────────────────────────

    def _hypothesis_amendment(self, memory: List[Tuple[DAG, float]]) -> DAG:
        memory_str = json.dumps(
            [{"dag": dag, "fitting_score": score} for dag, score in memory],
            indent=2,
        )
        # Reconstruct the set of all disease nodes from the latest DAG
        latest_dag = memory[-1][0]
        all_diseases = list(latest_dag.keys())

        prompt = (
            "Adjust the causal graph based on the current and previous versions "
            "stored in Memory, along with their fitting scores. "
            "Consider the following questions:\n"
            "- Are there any links that should be added?\n"
            "- Should any existing links be removed?\n"
            "- Should any directions be reversed?\n"
            "Generate a revised causal graph and output it in a valid JSON format.\n\n"
            f"Memory:\n{memory_str}\n\n"
            "Output a JSON object where each key is a disease name and its value "
            "is a list of its parent diseases (causes). Include all diseases."
        )
        return self._parse_dag(self._call_llm(prompt), all_diseases)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _parse_dag(self, llm_output: str, disease_names: List[str]) -> DAG:
        """Parse LLM JSON output into a DAG dict, with graceful fallback."""
        # Strip markdown code fences if present
        text = llm_output.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            dag = json.loads(text)
            if isinstance(dag, dict):
                # Ensure all diseases are present
                for d in disease_names:
                    if d not in dag:
                        dag[d] = []
                return dag
        except json.JSONDecodeError:
            pass

        # Fallback: empty DAG
        return {d: [] for d in disease_names}

    @staticmethod
    def _dags_equal(dag1: DAG, dag2: DAG) -> bool:
        if set(dag1.keys()) != set(dag2.keys()):
            return False
        for k in dag1:
            if sorted(dag1[k]) != sorted(dag2.get(k, [])):
                return False
        return True

    def _call_llm(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
