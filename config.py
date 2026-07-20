from pathlib import Path


APP_TITLE = "PrismicCipherAI"
APP_ICON = "✨"

GEMINI_MODEL = "gemini-1.5-flash"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

ENSEMBLE_TOP_K = 10
FINAL_CONTEXT_CHUNKS = 4
FAISS_FETCH_K = 24

VECTORSTORE_DIR = Path(".vectorstore")
