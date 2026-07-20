# Architectural Changes & Modernization Summary

This document summarizes every major architectural modernization and cleanup change made to the PrismicCipherAI project during the Tier 1 modernization pass and subsequent bugfix pass, relative to the original implementation. Each change is detailed below along with an explanation of why it improves the project's performance, reliability, and user experience.

---

## Tier 1 Modernization Pass

### 1. Improved Text Chunking (`CharacterTextSplitter` → `RecursiveCharacterTextSplitter`)
- **Change**: Replaced the basic `CharacterTextSplitter` with `RecursiveCharacterTextSplitter` (1000-character chunks with a 200-character overlap).
- **Why it improves the project**: `RecursiveCharacterTextSplitter` splits text hierarchically across natural boundaries (`\n\n`, `\n`, space, character) rather than purely at single delimiters. This prevents awkward mid-sentence breaks, preserves semantic structure, and ensures context continuity across chunks via overlapping segments.

### 2. LCEL-Based History-Aware Retrieval & Generation Chain (`ConversationalRetrievalChain` & `ConversationBufferMemory` Removed)
- **Change**: Replaced the legacy `ConversationalRetrievalChain` and `ConversationBufferMemory` with a custom, hand-rolled LangChain Expression Language (LCEL) chain (`rag/chain.py`).
- **Why it improves the project**: 
  - **Modernization & Deprecation Avoidance**: Legacy chain classes and memory components (`ConversationalRetrievalChain`, `ConversationBufferMemory`) are deprecated in modern LangChain (`langchain-core` / `langchain`). Furthermore, utility functions like `create_history_aware_retriever` and `create_retrieval_chain` have been moved to the legacy `langchain_classic` package. This project explicitly avoids `langchain_classic` in order to rely strictly on modern `langchain-core` primitives and maintain long-term compatibility.
  - **Transparency & Control**: Hand-rolling the history contextualization and question-answering pipeline via LCEL makes the exact prompt flow, state transformations, and input/output schema completely transparent and customizable.

### 3. Hybrid Retrieval (`FAISS` + `BM25`) with `MMR` Applied at the `FAISS` Sub-step
- **Change**: Upgraded single-vector search to a hybrid retrieval pipeline combining semantic vector search (`FAISS`) and exact-keyword search (`BM25Retriever`). Max Marginal Relevance (`MMR`) is applied at the `FAISS` vector search step (`fetch_k=20` candidate pool trimmed down based on diversity).
- **Why it improves the project**: Vector search excels at capturing conceptual meaning and paraphrases, while keyword retrieval (`BM25`) accurately locates exact terms, proper nouns, numerical codes, and section titles. Applying `MMR` ensures that the retrieved candidates cover diverse parts of the text rather than returning redundant or near-duplicate paragraphs.

### 4. Cross-Encoder Reranking (10 Candidates → Top 4)
- **Change**: Introduced a local cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to rerank the top 10 merged candidates from the hybrid search down to the top 4 most relevant chunks before passing them to the LLM.
- **Why it improves the project**: While bi-encoder vector embeddings (`sentence-transformers/all-MiniLM-L6-v2`) are fast for initial candidate retrieval, cross-encoders score the exact interaction between the query string and chunk text simultaneously. Reranking dramatically increases precision and reduces hallucination risk by ensuring only the highest-quality context reaches the context window of Gemini.

### 5. Source Attribution via Filename and Page Metadata
- **Change**: Enhanced document loading (`rag/loader.py`) and splitting (`rag/splitter.py`) to extract and preserve exact source metadata (`filename` and `page`) for every extracted chunk.
- **Why it improves the project**: Users can trace every fact or statement in the chatbot's answers back to the specific uploaded PDF and exact page number. This transparency builds trust and makes the chatbot practical for research and compliance verification.

### 6. Explicit "I Don't Know" Grounding Instruction in Prompts
- **Change**: Updated the answering prompt template (`rag/prompts.py`) to explicitly instruct the model (`Gemini`) to answer strictly based on the retrieved context, and to explicitly state when the answer cannot be found in the context.
- **Why it improves the project**: Prevents the LLM from relying on prior training data or hallucinating answers when the uploaded documents do not contain the requested information.

### 7. Persistent Vector Storage Keyed by Content Hash
- **Change**: Implemented automatic hashing (`SHA-256`) of uploaded PDF contents (`rag/loader.py`, `rag/vectorstore.py`). When files are processed, the `FAISS` index and document sidecar (used for re-initializing `BM25`) are saved to `.vectorstore/<hash>/`.
- **Why it improves the project**: Eliminates the overhead of re-reading and re-embedding large PDFs across sessions or restarts when the user uploads identical documents.

### 8. Streaming Responses
- **Change**: Configured Gemini for real-time token-by-token streaming right into the Streamlit chat interface (`app.py`).
- **Why it improves the project**: Significantly improves perceived latency. Instead of waiting several seconds for the entire response to finish generating, users receive immediate visual feedback as the answer is formed.

---

## Targeted Bugfix & Cleanup Pass

### 9. Automatic Model Download Fixes (`local_files_only` Removed)
- **Change**: Removed `model_kwargs={"local_files_only": True}` when initializing `HuggingFaceEmbeddings` (`rag/embeddings.py`) and removed `local_files_only=True` when initializing `CrossEncoder` (`rag/retriever.py`).
- **Why it improves the project**: Previously, if a user ran the project on a clean system or environment where `sentence-transformers/all-MiniLM-L6-v2` or `cross-encoder/ms-marco-MiniLM-L-6-v2` had not yet been downloaded to the local Hugging Face cache, model initialization would fail immediately with a local file check error. Removing this flag allows the models to be downloaded automatically from Hugging Face upon first use, while still leveraging the local disk cache automatically on all subsequent runs.

### 10. `CrossEncoder` In-Memory Caching (`_get_cross_encoder`)
- **Change**: Wrapped `CrossEncoder` initialization in `rag/retriever.py` with a module-level `functools.lru_cache` factory function (`_get_cross_encoder`).
- **Why it improves the project**: Previously, `HybridRerankRetriever._get_relevant_documents` instantiated a new `CrossEncoder` object from scratch on every single question submitted by the user. Loading weights from disk into memory on every turn added multi-second overhead per query. Caching the `CrossEncoder` instance at the module level ensures the model is loaded into memory only once per process, drastically speeding up retrieval time across multiple questions.

### 11. Log File Cleanup & `.gitignore` Update
- **Change**: Deleted committed runtime log files (`streamlit.out.log`, `streamlit.err.log`) from the repository and updated `.gitignore` to exclude `*.log`, `streamlit.out.log`, and `streamlit.err.log`.
- **Why it improves the project**: Prevents local execution logs and machine-specific diagnostic outputs from polluting version control and repository diffs.
