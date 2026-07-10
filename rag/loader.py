from __future__ import annotations

import hashlib
from io import BytesIO
from dataclasses import dataclass
from typing import BinaryIO, Iterable

from langchain_core.documents import Document
from pypdf import PdfReader


@dataclass(frozen=True)
class UploadedPdf:
    """PDF bytes and display name captured from Streamlit uploads."""

    name: str
    data: bytes


def read_uploaded_pdfs(pdf_docs: Iterable[BinaryIO]) -> list[UploadedPdf]:
    """Read uploaded PDF files once so hashing and parsing use identical bytes."""
    uploads: list[UploadedPdf] = []

    for pdf in pdf_docs:
        if hasattr(pdf, "getvalue"):
            data = pdf.getvalue()
        else:
            data = pdf.read()

        name = getattr(pdf, "name", "uploaded.pdf")
        uploads.append(UploadedPdf(name=name, data=data))

    return uploads


def hash_uploads(uploads: Iterable[UploadedPdf]) -> str:
    """Create a stable SHA-256 hash from the uploaded file names and bytes."""
    digest = hashlib.sha256()

    for upload in sorted(uploads, key=lambda item: item.name):
        digest.update(upload.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(upload.data)
        digest.update(b"\0")

    return digest.hexdigest()


def load_pdf_documents(uploads: Iterable[UploadedPdf]) -> list[Document]:
    """Extract page-level PDF text while preserving filename and page metadata."""
    documents: list[Document] = []

    for upload in uploads:
        try:
            reader = PdfReader(BytesIO(upload.data))
        except Exception as exc:
            raise ValueError(f"Could not read {upload.name}. Please upload a valid PDF.") from exc

        for page_index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""

            text = text.strip()
            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={"filename": upload.name, "page": page_index},
                )
            )

    if not documents:
        raise ValueError("No extractable text was found in the uploaded PDFs.")

    return documents
