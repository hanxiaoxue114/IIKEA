# II-KEA: Interpretable and Interactable Predictive Healthcare with Knowledge-Enhanced Agentic Causal Discovery

> **No Black Boxes: Interpretable and Interactable Predictive Healthcare with Knowledge-Enhanced Agentic Causal Discovery**
> Xiaoxue Han, Pengfei Hu, Chang Lu, Jun-En Ding, Feng Liu, Yue Ning
> Stevens Institute of Technology
> arXiv:2505.16288

---

## Overview

Deep learning models trained on Electronic Health Records (EHR) achieve strong predictive accuracy but function as "black boxes" — they lack interpretability and do not allow clinicians to interact with or customize their reasoning. **II-KEA** addresses both limitations through a multi-agent framework powered by Large Language Models (LLMs) and causal discovery.

II-KEA answers not just *"What diseases will this patient likely develop?"* but the deeper causal question: *"What diseases are likely to be caused by the conditions a patient has already been diagnosed with?"*

---

## Key Features

- **Interpretable** — every prediction comes with a causal graph and a detailed natural language explanation of the reasoning process
- **Interactable** — clinicians can inject their own domain knowledge and preferences via custom prompts or a clinician comment field
- **Knowledge-grounded** — retrieval-augmented generation (RAG) over a medical Wikipedia knowledge base enriches causal reasoning
- **Training-free** — II-KEA requires no model training; it uses data preprocessing and LLM inference only

---

## Framework

II-KEA consists of three collaborative LLM agents:

### 1. Knowledge Synthesis Agent (A_knowledge)
Retrieves and summarises the most relevant external medical knowledge for a patient.

- Generates a targeted search query from the patient's diagnosis history and candidate diseases
- Retrieves the top-*k* most semantically similar documents from a ChromaDB vector database
- Performs "reasoning-in-documents": summarises each document to retain only information relevant to the patient

### 2. Causal Discovery Agent (A_causal)
Uncovers causal relationships among diseases through an iterative loop:

1. **Hypothesis generation** — LLM produces an initial causal DAG in JSON format, using the knowledge summaries as context
2. **Model fitting** — log-likelihood of the DAG is computed against the observational Diagnosis Matrix A_D (Eq. 5)
3. **Post-processing** — the current DAG and its score are stored in memory
4. **Hypothesis amendment** — LLM refines the DAG by reviewing its memory of previous graphs and scores

This loop repeats until the graph converges or a maximum number of iterations is reached.

### 3. Decision-Making Agent (A_dm)
Integrates all available information to produce the final prediction:

- Diagnosis history, candidate diseases, knowledge summaries, causal graph
- Optional clinician comment for personalised predictions
- Outputs a ranked list of predicted ICD-9 codes and a detailed causal explanation

---

## Clinical Datasets

II-KEA is evaluated on two benchmark EHR datasets:

| Dataset   | Patients | Visit Period  | Train  | Val   | Test  |
|-----------|----------|---------------|--------|-------|-------|
| MIMIC-III | 7,493    | 2001 – 2012   | 6,000  | 1,900 | 1,000 |
| MIMIC-IV  | 10,000*  | 2013 – 2019   | 8,000  | 1,000 | 1,000 |

\* 10,000 randomly sampled from 85,155 patients to minimise overlap with MIMIC-III.

Both datasets are available via [PhysioNet](https://physionet.org) under a credentialed data use agreement.

---

## Results

II-KEA achieves state-of-the-art performance on diagnosis prediction, outperforming all baselines including transformer-based and graph-based models.

**MIMIC-III**

| Model       | w-F1  | R@10  | R@20  |
|-------------|-------|-------|-------|
| GT-BEHRT    | 25.21 | 36.15 | 40.97 |
| GraphCare   | 25.16 | 36.74 | 41.89 |
| RAM-EHR     | 23.27 | 34.66 | 38.49 |
| DualMAR     | 25.37 | 38.24 | 41.86 |
| **II-KEA**  | **28.61** | **38.52** | **43.86** |

**MIMIC-IV**

| Model       | w-F1  | R@10  | R@20  |
|-------------|-------|-------|-------|
| GT-BEHRT    | 30.17 | 44.93 | 50.67 |
| GraphCare   | 27.59 | 42.07 | 48.19 |
| RAM-EHR     | 26.97 | 41.17 | 46.23 |
| DualMAR     | 27.97 | 44.07 | 48.19 |
| **II-KEA**  | **29.87** | **45.66** | **51.73** |

---

## Repository Structure

```
code/
├── config.py                          # LLM, ChromaDB, and hyperparameter settings
├── requirements.txt                   # Python dependencies
├── main.py                            # Full inference pipeline (Algorithm 4)
├── evaluate.py                        # w-F1 and R@k evaluation metrics
├── data/
│   └── preprocess.py                  # Build A_T and A_D from MIMIC data
├── knowledge/
│   ├── build_db.py                    # Scrape Wikipedia and build ChromaDB
│   └── vector_store.py                # ChromaDB + Sentence-BERT wrapper
└── agents/
    ├── knowledge_synthesis_agent.py   # Algorithm 1
    ├── causal_discovery_agent.py      # Algorithm 2
    └── decision_making_agent.py       # Algorithm 3
```

---

## Setup

### Requirements

- Python 3.10+
- Azure OpenAI access (GPT-4o mini)
- MIMIC-III or MIMIC-IV access via PhysioNet

### Installation

```bash
pip install -r requirements.txt
```

### Environment Variables

```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

---

## Usage

### Step 1 — Preprocess EHR Data

Builds the Disease Transition Probability Matrix (A_T) and Diagnosis Matrix (A_D) from MIMIC.

```bash
python -m data.preprocess \
  --diagnoses  /path/to/DIAGNOSES_ICD.csv \
  --admissions /path/to/ADMISSIONS.csv \
  --out_dir    data/processed \
  --dataset    mimic3
```

### Step 2 — Build Knowledge Database

Scrapes Wikipedia for each ICD-9 disease and indexes documents in ChromaDB.

```bash
python -m knowledge.build_db \
  --disease_list data/processed/disease_list.pkl
```

### Step 3a — Single-Patient Demo

```bash
python main.py \
  --diagnoses  /path/to/DIAGNOSES_ICD.csv \
  --admissions /path/to/ADMISSIONS.csv \
  --single_patient 401 250 428 585 \
  --clinician_comment "Patient is particularly concerned about kidney-related diseases."
```

### Step 3b — Full Dataset Evaluation

```bash
python main.py \
  --diagnoses  /path/to/DIAGNOSES_ICD.csv \
  --admissions /path/to/ADMISSIONS.csv \
  --dataset    mimic3 \
  --output     results.json
```

---

## Implementation Details

| Component         | Detail                                      |
|-------------------|---------------------------------------------|
| LLM               | ChatGPT-4o mini via Azure OpenAI            |
| LLM temperature   | 0                                           |
| Embeddings        | `all-MPNet-base-v2` (Sentence-BERT)         |
| Vector database   | ChromaDB                                    |
| Knowledge source  | Wikipedia (ICD-9 disease pages)             |
| Causal iterations | Up to 5 (stops early if DAG converges)      |
| Transition threshold ε | 0.01                                   |
| Evaluation runs   | Average over 5 runs (paper reports std dev) |

---

## Citation

```bibtex
@article{han2025iikea,
  title     = {No Black Boxes: Interpretable and Interactable Predictive Healthcare
               with Knowledge-Enhanced Agentic Causal Discovery},
  author    = {Han, Xiaoxue and Hu, Pengfei and Lu, Chang and Ding, Jun-En
               and Liu, Feng and Ning, Yue},
  journal   = {arXiv preprint arXiv:2505.16288},
  year      = {2025}
}
```

---

## License

This project is for research purposes. Use of MIMIC data requires credentialed access via PhysioNet and compliance with the PhysioNet Credentialed Health Data Use Agreement 1.5.0. All interactions with EHR data must be conducted through compliant services (e.g., Azure OpenAI) per the responsible use guidelines.
