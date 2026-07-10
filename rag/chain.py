from __future__ import annotations

from collections.abc import Iterable

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

from config import LLM_MODEL
from rag.prompts import contextualize_prompt, qa_prompt
from rag.retriever import HybridRerankRetriever


def create_llm(streaming: bool = False) -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, temperature=0, streaming=streaming)


def create_history_aware_retriever(llm: ChatOllama, retriever: HybridRerankRetriever):
    """LCEL history-aware retriever equivalent for LangChain 1.x."""

    def retrieve(inputs: dict) -> list[Document]:
        chat_history = inputs.get("chat_history") or []
        user_input = inputs["input"]

        if chat_history:
            query = (contextualize_prompt | llm | StrOutputParser()).invoke(
                {"input": user_input, "chat_history": chat_history}
            )
        else:
            query = user_input

        return retriever.invoke(query)

    return RunnableLambda(retrieve)


def create_retrieval_chain(history_aware_retriever, llm: ChatOllama):
    """LCEL retrieval chain that returns answer text plus source documents."""

    def format_context(inputs: dict) -> str:
        return format_documents(inputs.get("context", []))

    answer_chain = (
        RunnablePassthrough.assign(context=format_context)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return RunnablePassthrough.assign(context=history_aware_retriever).assign(answer=answer_chain)


def build_rag_chain(retriever: HybridRerankRetriever, streaming: bool = False):
    llm = create_llm(streaming=streaming)
    history_aware_retriever = create_history_aware_retriever(llm, retriever)
    return create_retrieval_chain(history_aware_retriever, llm)


def format_documents(documents: Iterable[Document]) -> str:
    formatted = []
    for doc in documents:
        filename = doc.metadata.get("filename", "Unknown file")
        page = doc.metadata.get("page", "Unknown page")
        formatted.append(f"Source: {filename}, page {page}\n{doc.page_content}")
    return "\n\n".join(formatted)


def to_langchain_history(chat_history: list[dict[str, str]]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in chat_history:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            messages.append(AIMessage(content=item["content"]))
    return messages
