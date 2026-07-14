import importlib
import streamlit as st
from dotenv import load_dotenv

from config import APP_ICON, APP_TITLE
from htmlTemplates import bot_template, css, user_template
from rag.chain import build_rag_chain, to_langchain_history
import rag.loader
import rag.vectorstore
import rag.retriever
from rag.loader import hash_uploads, load_pdf_documents, read_uploaded_pdfs, sanitize_text
from rag.retriever import HybridRerankRetriever
from rag.splitter import split_documents
from rag.vectorstore import build_or_load_vectorstore


def initialize_session_state() -> None:
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "sources" not in st.session_state:
        st.session_state.sources = []


def render_chat_history() -> None:
    for message in st.session_state.chat_history:
        template = user_template if message["role"] == "user" else bot_template
        st.write(template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)


def render_sources(sources) -> None:
    if not sources:
        return

    seen: set[tuple[str, int | str]] = set()
    unique_sources = []
    for doc in sources:
        filename = doc.metadata.get("filename", "Unknown file")
        page = doc.metadata.get("page", "Unknown page")
        key = (filename, page)
        if key not in seen:
            seen.add(key)
            unique_sources.append(key)

    if unique_sources:
        st.caption("Sources")
        for filename, page in unique_sources:
            st.caption(f"{filename} - Page {page}")


def process_documents(pdf_docs) -> None:
    try:
        importlib.reload(rag.loader)
        importlib.reload(rag.vectorstore)
        importlib.reload(rag.retriever)

        uploads = read_uploaded_pdfs(pdf_docs)
        content_hash = hash_uploads(uploads)
        documents = load_pdf_documents(uploads)
        chunks = split_documents(documents)

        if not chunks:
            raise ValueError("No text chunks could be created from the uploaded PDFs.")

        for chunk in chunks:
            if hasattr(chunk, "page_content"):
                chunk.page_content = sanitize_text(chunk.page_content)

        vectorstore, saved_chunks, loaded_from_cache = build_or_load_vectorstore(
            chunks=chunks,
            content_hash=content_hash,
        )
    except RuntimeError as exc:
        if "client has been closed" in str(exc):
            raise RuntimeError(
                "The embedding model could not be loaded from Hugging Face. "
                "Check your internet connection, proxy/VPN settings, or pre-download "
                "sentence-transformers/all-MiniLM-L6-v2 before processing PDFs."
            ) from exc
        raise

    retriever = HybridRerankRetriever(
        vectorstore=vectorstore,
        documents=saved_chunks,
    )
    st.session_state.rag_chain = build_rag_chain(retriever, streaming=True)
    st.session_state.chat_history = []
    st.session_state.sources = []

    if loaded_from_cache:
        st.success("Existing vector database loaded successfully!")
    else:
        st.success("Documents processed successfully!")


def handle_userinput(user_question: str) -> None:
    if st.session_state.rag_chain is None:
        st.warning("Please upload and process your PDFs first.")
        return

    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)

    history_messages = to_langchain_history(st.session_state.chat_history[:-1])
    chain_input = {"input": user_question, "chat_history": history_messages}

    answer_placeholder = st.empty()
    answer = ""
    sources = []

    try:
        for chunk in st.session_state.rag_chain.stream(chain_input):
            if "context" in chunk:
                sources = chunk["context"]
            if "answer" in chunk:
                answer += chunk["answer"]
                answer_placeholder.write(
                    bot_template.replace("{{MSG}}", answer),
                    unsafe_allow_html=True,
                )
    except Exception as exc:
        st.error(
            "The local model could not generate a response. "
            "Make sure Ollama is running and llama3.2 is installed."
        )
        st.exception(exc)
        st.session_state.chat_history.pop()
        return

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.session_state.sources = sources
    render_sources(sources)


def main() -> None:
    load_dotenv()

    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON)
    st.write(css, unsafe_allow_html=True)
    initialize_session_state()

    st.header("PrismicCipherAI - Chat with Multiple PDFs 📑")

    render_chat_history()

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")

        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing..."):
                    try:
                        process_documents(pdf_docs)
                    except ValueError as exc:
                        st.warning(str(exc))
                    except Exception as exc:
                        st.error("Unable to process the uploaded PDFs.")
                        st.exception(exc)


if __name__ == "__main__":
    main()
