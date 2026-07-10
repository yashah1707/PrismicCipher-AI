from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


CONTEXTUALIZE_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question, rewrite the question "
    "as a standalone search query. Do not answer the question."
)

QA_SYSTEM_PROMPT = (
    "You are PrismicCipherAI, a local assistant that answers questions using "
    "only the retrieved PDF context below.\n\n"
    "If the answer is not contained in the retrieved context, respond exactly: "
    "\"I don't know based on the uploaded documents.\"\n\n"
    "Never fabricate information. Keep answers concise and factual.\n\n"
    "Context:\n{context}"
)


contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
