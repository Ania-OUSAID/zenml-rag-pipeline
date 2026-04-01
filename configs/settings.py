"""Configuration centralisée du projet RAG ZenML."""

import os

from dotenv import load_dotenv

load_dotenv()

# ── Chemins ──────────────────────────────────────────────────
PDF_DIR = os.getenv("PDF_DIR", "./pdfs")

# ── Chunking ─────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

# ── Embedding ────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

# ── ChromaDB ─────────────────────────────────────────────────
CHROMA_COLLECTION = "rag_tp_collection"
CHROMA_PERSIST_DIR = "./chroma_db"

# ── LLM (Groq) ──────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = "llama-3.3-70b-versatile"
# LLM_MODEL = "openai/gpt-oss-20b"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 1024

# ── Retrieval ────────────────────────────────────────────────
DEFAULT_TOP_K = 5
DEFAULT_BM25_WEIGHT = 0.4

# ── RAGAS (Ollama) ─────────────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


# ── Prompt système ───────────────────────────────────────────
SYSTEM_PROMPT = """Tu es un assistant expert en intelligence artificielle.
Tu réponds aux questions en te basant sur le contexte fourni ci-dessous.

Règles :
- Utilise les informations du contexte seulement pour construire ta réponse clairement en français.
- Si le contexte contient une réponse partielle, donne ce que tu peux et précise que l'information est incomplète.
- Si le contexte ne contient AUCUNE information pertinente, dis-le clairement.
- Ne fais JAMAIS référence au "contexte" dans ta réponse.
- Cite des éléments précis du texte pour appuyer ta réponse.

Contexte :
{context}"""
