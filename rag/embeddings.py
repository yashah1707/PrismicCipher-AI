from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the configured local embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
    )
