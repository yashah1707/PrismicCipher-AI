from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config import VECTORSTORE_DIR
from rag.embeddings import get_embeddings
from rag.loader import sanitize_text

DOCSTORE_FILE = "documents.json"


def index_path(content_hash: str) -> Path:
    return VECTORSTORE_DIR / content_hash


def has_saved_index(content_hash: str) -> bool:
    path = index_path(content_hash)
    return path.exists() and (path / DOCSTORE_FILE).exists()


def build_or_load_vectorstore(
    chunks: list[Document],
    content_hash: str,
) -> tuple[FAISS, list[Document], bool]:
    """Load an existing FAISS index or build and persist a new one."""
    embeddings = get_embeddings()
    path = index_path(content_hash)

    if has_saved_index(content_hash):
        vectorstore = FAISS.load_local(
            str(path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vectorstore, load_saved_documents(path), True

    for doc in chunks:
        doc.page_content = sanitize_text(doc.page_content)

    path.mkdir(parents=True, exist_ok=True)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(path))
    save_documents(path, chunks)
    return vectorstore, chunks, False


def save_documents(path: Path, documents: list[Document]) -> None:
    payload = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in documents
    ]
    (path / DOCSTORE_FILE).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_saved_documents(path: Path) -> list[Document]:
    payload = json.loads((path / DOCSTORE_FILE).read_text(encoding="utf-8"))
    return [
        Document(page_content=sanitize_text(item["page_content"]), metadata=item.get("metadata", {}))
        for item in payload
    ]
