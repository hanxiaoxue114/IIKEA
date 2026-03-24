"""
Decision-Making Agent  (A_dm)  — Algorithm 3 in the paper.

Integrates:
  - Diagnosis history D_p
  - Candidate diseases S_p
  - Summarised knowledge Gamma_p^summary
  - Causal graph G^s
  - Optional clinician comment C

Outputs predicted ICD-9 codes and a detailed explanation.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from openai import AzureOpenAI

from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_DEPLOYMENT_NAME,
    LLM_TEMPERATURE,
)
from agents.causal_discovery_agent import DAG


class DecisionMakingAgent:
    """
    A_dm: predicts future diagnoses and provides causal explanations.
    """

    def __init__(self) -> None:
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
        causal_graph: DAG,
        clinician_comment: Optional[str] = None,
    ) -> Tuple[List[str], str]:
        """
        Returns (predicted_icd9_codes, explanation_text).
        """
        summary_text  = "\n\n".join(summaries)
        dag_json      = json.dumps(causal_graph, indent=2)
        comment       = clinician_comment or ""

        prompt = self._build_prompt(
            diagnosis_history, candidate_diseases,
            summary_text, dag_json, comment,
        )
        raw = self._call_llm(prompt)
        return self._parse_output(raw)

    # ── Prompt (Appendix C) ────────────────────────────────────────────────

    def _build_prompt(
        self,
        diagnosis_history: List[str],
        candidate_diseases: List[str],
        summary: str,
        dag_json: str,
        clinician_comment: str,
    ) -> str:
        history_str = ", ".join(diagnosis_history)
        candidate_str = ", ".join(candidate_diseases)

        return (
            "Predict a list of diseases the patient may be diagnosed with in the "
            "future based on:\n\n"
            f"Patient diagnosis history: {history_str}\n\n"
            f"Candidate diseases: {candidate_str}\n\n"
            f"Patient summary and disease information:\n{summary}\n\n"
            f"Causal DAG of disease relationships:\n{dag_json}\n\n"
            f"Optional clinician comment: {clinician_comment}\n\n"
            "Output format:\n"
            "1. A JSON list of predicted ICD-9 codes.\n"
            "2. A detailed explanation of the reasoning process.\n"
            "Separate the two parts using the special token <SEP>.\n\n"
            "Example output:\n"
            '[\"401.9\", \"250.00\", \"585.3\"]\n'
            "<SEP>\n"
            "The patient has a history of hypertension and diabetes, which causally "
            "increases the risk of chronic kidney disease as evidenced by the causal graph..."
        )

    # ── Output parsing ─────────────────────────────────────────────────────

    def _parse_output(self, raw: str) -> Tuple[List[str], str]:
        """
        Split on <SEP>, parse JSON codes from part 1, return explanation from part 2.
        """
        if "<SEP>" in raw:
            json_part, explanation = raw.split("<SEP>", 1)
        else:
            json_part   = raw
            explanation = ""

        # Extract JSON list from the first part
        codes: List[str] = []
        json_part = json_part.strip()
        # Find first [...] block
        match = re.search(r"\[.*?\]", json_part, re.DOTALL)
        if match:
            try:
                codes = json.loads(match.group())
                if not isinstance(codes, list):
                    codes = []
            except json.JSONDecodeError:
                codes = []

        return [str(c).strip() for c in codes], explanation.strip()

    # ── LLM helper ─────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
