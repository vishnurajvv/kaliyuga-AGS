"""
rag_engine.py
=============
LangChain-based RAG layer. Pulls rows from the Persona sheet and the
Company Details sheet, turns each row into a Document, embeds them with a
local Ollama embedding model, and serves similarity-search retrieval to
the orchestrator on every customer turn.

Two separate retrievers are kept:
  - persona_retriever  -> tone/behavior/character guidance
  - company_retriever   -> factual company/product/policy knowledge

The index is rebuilt on a timer (RAG_REFRESH_INTERVAL_SECONDS) so edits
made directly in Google Sheets show up without restarting the service.
"""

import logging
import threading
import time
from functools import lru_cache

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from . import config
from .sheets_client import get_sheets_client

logger = logging.getLogger("proto_bpo.rag_engine")


def _rows_to_documents(rows: list[dict], source: str) -> list[Document]:
    docs = []
    for i, row in enumerate(rows):
        # Concatenate all columns into one readable text block; keep the
        # raw dict as metadata so the admin panel / debugging can inspect it.
        text = "\n".join(f"{k}: {v}" for k, v in row.items() if str(v).strip())
        if not text.strip():
            continue
        docs.append(Document(page_content=text, metadata={"source": source, "row": i}))
    return docs


class RAGEngine:
    def __init__(self):
        self._embeddings = OllamaEmbeddings(
            model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL
        )
        self._persona_store: Chroma | None = None
        self._company_store: Chroma | None = None
        self._lock = threading.Lock()
        self._refresh_thread: threading.Thread | None = None
        self.build_index()

    # -- index construction ---------------------------------------------

    def build_index(self) -> None:
        with self._lock:
            client = get_sheets_client()

            persona_rows = client.get_persona_rules()
            company_rows = client.get_company_details()

            persona_docs = _rows_to_documents(persona_rows, source="persona")
            company_docs = _rows_to_documents(company_rows, source="company")

            logger.info(
                "Rebuilding RAG index: %d persona rows, %d company rows",
                len(persona_docs), len(company_docs),
            )

            if persona_docs:
                self._persona_store = Chroma.from_documents(
                    persona_docs,
                    self._embeddings,
                    collection_name="persona",
                    persist_directory=str(config.VECTOR_STORE_DIR / "persona"),
                )
            if company_docs:
                self._company_store = Chroma.from_documents(
                    company_docs,
                    self._embeddings,
                    collection_name="company",
                    persist_directory=str(config.VECTOR_STORE_DIR / "company"),
                )

    def start_auto_refresh(self) -> None:
        if self._refresh_thread and self._refresh_thread.is_alive():
            return

        def _loop():
            while True:
                time.sleep(config.RAG_REFRESH_INTERVAL_SECONDS)
                try:
                    self.build_index()
                except Exception:
                    logger.exception("Scheduled RAG index refresh failed")

        self._refresh_thread = threading.Thread(target=_loop, daemon=True)
        self._refresh_thread.start()
        logger.info(
            "RAG auto-refresh started (every %ss)", config.RAG_REFRESH_INTERVAL_SECONDS
        )

    # -- retrieval -----------------------------------------------------

    def get_persona_context(self, query: str, k: int = 3) -> str:
        if not self._persona_store:
            return ""
        results = self._persona_store.similarity_search(query, k=k)
        return "\n---\n".join(d.page_content for d in results)

    def get_company_context(self, query: str, k: int = 4) -> str:
        if not self._company_store:
            return ""
        results = self._company_store.similarity_search(query, k=k)
        return "\n---\n".join(d.page_content for d in results)

    def retrieve(self, query: str) -> dict[str, str]:
        """Convenience call used by the orchestrator on every turn."""
        return {
            "persona": self.get_persona_context(query),
            "company": self.get_company_context(query),
        }


@lru_cache(maxsize=1)
def get_rag_engine() -> RAGEngine:
    return RAGEngine()
