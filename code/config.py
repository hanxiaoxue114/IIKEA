import os

# ── LLM (Azure OpenAI) ─────────────────────────────────────────────────────
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = "2024-02-01"
AZURE_DEPLOYMENT_NAME = "gpt-4o-mini"
LLM_TEMPERATURE = 0

# ── Embeddings ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MPNet-base-v2"

# ── ChromaDB ───────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "medical_knowledge"

# ── Knowledge retrieval ────────────────────────────────────────────────────
TOP_K_DOCS = 5          # number of documents retrieved per query

# ── Candidate disease selection ────────────────────────────────────────────
TRANSITION_THRESHOLD = 0.01   # ε: minimum A_T value to include candidate

# ── Causal discovery ───────────────────────────────────────────────────────
MAX_CAUSAL_ITERATIONS = 5
CAUSAL_STOP_THRESHOLD = 0.0   # stop if DAG unchanged between iterations

# ── Evaluation ─────────────────────────────────────────────────────────────
RECALL_K_VALUES = [10, 20]

# ── Data splits ────────────────────────────────────────────────────────────
# MIMIC-III: train=6000, val=1900, test=1000
# MIMIC-IV:  train=8000, val=1000, test=1000
MIMIC3_SPLITS = {"train": 6000, "val": 1900, "test": 1000}
MIMIC4_SPLITS = {"train": 8000, "val": 1000, "test": 1000}
