"""
Data preprocessing for II-KEA.

Builds two matrices from MIMIC-III/IV EHR data:
  - A_T : Disease Transition Probability Matrix  (Eq. 1-2 in paper)
  - A_D : Diagnosis Matrix                       (Eq. 3 in paper)

MIMIC data files expected:
  - DIAGNOSES_ICD.csv  (SUBJECT_ID, HADM_ID, ICD9_CODE)
  - ADMISSIONS.csv     (SUBJECT_ID, HADM_ID, ADMITTIME)
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ── Types ──────────────────────────────────────────────────────────────────
PatientVisits = Dict[str, List[List[str]]]   # {patient_id: [[codes_v1], ...]}


# ── Loading ────────────────────────────────────────────────────────────────

def load_mimic(diagnoses_path: str, admissions_path: str) -> PatientVisits:
    """
    Load MIMIC-III or MIMIC-IV and return patient visit sequences sorted by
    admission time.  Only patients with >= 2 visits are kept.
    """
    diag = pd.read_csv(diagnoses_path, dtype=str)
    adm  = pd.read_csv(admissions_path, dtype=str, parse_dates=["ADMITTIME"])

    # Normalise column names (MIMIC-IV uses lower-case)
    diag.columns = diag.columns.str.upper()
    adm.columns  = adm.columns.str.upper()

    # Build hadm -> codes mapping
    hadm_codes: Dict[str, List[str]] = defaultdict(list)
    for _, row in diag.iterrows():
        code = str(row["ICD9_CODE"]).strip()
        if code and code != "nan":
            hadm_codes[row["HADM_ID"]].append(code)

    # Sort admissions per patient by time
    adm = adm.sort_values(["SUBJECT_ID", "ADMITTIME"])
    patients: PatientVisits = {}
    for pid, grp in adm.groupby("SUBJECT_ID"):
        visits = [hadm_codes[hid] for hid in grp["HADM_ID"] if hadm_codes[hid]]
        if len(visits) >= 2:
            patients[str(pid)] = visits

    return patients


def split_patients(
    patients: PatientVisits,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int = 42,
) -> Tuple[PatientVisits, PatientVisits, PatientVisits]:
    """Split patients into train / val / test sets."""
    rng = np.random.default_rng(seed)
    pids = list(patients.keys())
    rng.shuffle(pids)

    train_ids = pids[:train_size]
    val_ids   = pids[train_size : train_size + val_size]
    test_ids  = pids[train_size + val_size : train_size + val_size + test_size]

    return (
        {p: patients[p] for p in train_ids},
        {p: patients[p] for p in val_ids},
        {p: patients[p] for p in test_ids},
    )


# ── Disease Transition Probability Matrix  A_T  (Eq. 1-2) ─────────────────

def build_transition_matrix(
    patients: PatientVisits,
    disease_list: List[str],
) -> np.ndarray:
    """
    A_T[a, b] = N_ab / sum_{p in P} sum_{i=1}^{m_p - 1} 1[a in D_p^i]

    N_ab = count of (patient, visit i) where a in D_p^i AND
           (b in D_p^{i+1}  OR  b in D_p^i)
    """
    d2i = {d: i for i, d in enumerate(disease_list)}
    n   = len(disease_list)

    N_ab      = np.zeros((n, n), dtype=np.float64)
    denom_a   = np.zeros(n,     dtype=np.float64)

    for visits in patients.values():
        for t in range(len(visits) - 1):
            cur  = set(visits[t])
            nxt  = set(visits[t + 1])
            for a in cur:
                if a not in d2i:
                    continue
                ai = d2i[a]
                denom_a[ai] += 1
                for b in nxt | cur:
                    if b not in d2i:
                        continue
                    N_ab[ai, d2i[b]] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        A_T = np.where(denom_a[:, None] > 0, N_ab / denom_a[:, None], 0.0)

    return A_T


# ── Diagnosis Matrix  A_D  (Eq. 3) ────────────────────────────────────────

def build_diagnosis_matrix(
    patients: PatientVisits,
    disease_list: List[str],
) -> np.ndarray:
    """
    A_D[p, a] = 1  if patient p has ever been diagnosed with disease a
                    in any revisit (visit >= 2).
    Returns shape (num_patients, num_diseases).
    """
    d2i    = {d: i for i, d in enumerate(disease_list)}
    n_d    = len(disease_list)
    pids   = sorted(patients.keys())
    n_p    = len(pids)
    A_D    = np.zeros((n_p, n_d), dtype=np.int8)

    for pi, pid in enumerate(pids):
        visits = patients[pid]
        # Union of all diseases across revisits (visit index >= 1)
        revisit_diseases: set = set()
        for visit in visits[1:]:
            revisit_diseases.update(visit)
        for d in revisit_diseases:
            if d in d2i:
                A_D[pi, d2i[d]] = 1

    return A_D, pids


# ── Candidate disease selection  (Eq. 4) ──────────────────────────────────

def get_candidate_diseases(
    diagnosis_history: List[str],
    A_T: np.ndarray,
    disease_list: List[str],
    threshold: float = 0.01,
) -> List[str]:
    """
    S^p = { b | A_T[a, b] > threshold,  for any a in D_p }
    """
    d2i = {d: i for i, d in enumerate(disease_list)}
    candidates: set = set()
    for a in diagnosis_history:
        if a not in d2i:
            continue
        row = A_T[d2i[a]]
        for j, val in enumerate(row):
            if val > threshold:
                candidates.add(disease_list[j])
    # Exclude diseases already diagnosed
    candidates -= set(diagnosis_history)
    return sorted(candidates)


# ── Persistence ────────────────────────────────────────────────────────────

def save_matrices(
    A_T: np.ndarray,
    A_D: np.ndarray,
    patient_ids: List[str],
    disease_list: List[str],
    out_dir: str,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "A_T.npy", A_T)
    np.save(out / "A_D.npy", A_D)
    with open(out / "patient_ids.pkl", "wb") as f:
        pickle.dump(patient_ids, f)
    with open(out / "disease_list.pkl", "wb") as f:
        pickle.dump(disease_list, f)
    print(f"Saved matrices to {out}/")


def load_matrices(out_dir: str):
    out = Path(out_dir)
    A_T = np.load(out / "A_T.npy")
    A_D = np.load(out / "A_D.npy")
    with open(out / "patient_ids.pkl", "rb") as f:
        patient_ids = pickle.load(f)
    with open(out / "disease_list.pkl", "rb") as f:
        disease_list = pickle.load(f)
    return A_T, A_D, patient_ids, disease_list


# ── CLI convenience ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from config import MIMIC3_SPLITS, TRANSITION_THRESHOLD

    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnoses", required=True, help="Path to DIAGNOSES_ICD.csv")
    parser.add_argument("--admissions", required=True, help="Path to ADMISSIONS.csv")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--dataset", choices=["mimic3", "mimic4"], default="mimic3")
    args = parser.parse_args()

    print("Loading MIMIC data...")
    patients = load_mimic(args.diagnoses, args.admissions)
    print(f"  Total patients with >= 2 visits: {len(patients)}")

    splits = MIMIC3_SPLITS
    train_p, val_p, test_p = split_patients(
        patients, splits["train"], splits["val"], splits["test"]
    )
    print(f"  Train: {len(train_p)}  Val: {len(val_p)}  Test: {len(test_p)}")

    # Collect all diseases from training set
    disease_set: set = set()
    for visits in train_p.values():
        for visit in visits:
            disease_set.update(visit)
    disease_list = sorted(disease_set)
    print(f"  Unique diseases (train): {len(disease_list)}")

    print("Building A_T...")
    A_T = build_transition_matrix(train_p, disease_list)

    print("Building A_D...")
    A_D, patient_ids = build_diagnosis_matrix(train_p, disease_list)

    save_matrices(A_T, A_D, patient_ids, disease_list, args.out_dir)
    print("Done.")
