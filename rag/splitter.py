from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SEPARATORS, CHUNK_SIZE


def split_documents(documents: list[Document]) -> list[Document]:
    """Split page-level documents into overlapping chunks with metadata intact."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        length_function=len,
    )
    return splitter.split_documents(documents)
