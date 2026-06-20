import logging
import re
from typing import Any, Dict, List
from app.db.chroma_client import chroma_client
from app.services.embeddings import embeddings_service
from app.config import settings

logger = logging.getLogger(__name__)

SPACE_PATTERN = re.compile(r"\s+")


class RAGService:
    initialized = False
    default_chunk_words = 180
    default_chunk_overlap = 35

    @classmethod
    def initialize(cls):
        try:
            chroma_client.initialize()
            embeddings_service.initialize()
            cls.initialized = True
            logger.info("RAG retrieval service initialized")
        except Exception as e:
            logger.warning("RAG service initialization failed: %s", str(e))
            cls.initialized = False

    @classmethod
    def normalize_text(cls, text: str) -> str:
        return SPACE_PATTERN.sub(" ", text or "").strip()

    @classmethod
    def chunk_text(
        cls,
        text: str,
        chunk_words: int = None,
        overlap_words: int = None,
    ) -> List[str]:
        chunk_words = chunk_words or cls.default_chunk_words
        overlap_words = overlap_words if overlap_words is not None else cls.default_chunk_overlap
        text = cls.normalize_text(text)
        words = text.split()
        if not words:
            return []
        if len(words) <= chunk_words:
            return [text]

        chunks = []
        start = 0
        step = max(chunk_words - overlap_words, 1)
        while start < len(words):
            chunk = " ".join(words[start:start + chunk_words]).strip()
            if chunk:
                chunks.append(chunk)
            start += step
        return chunks

    @classmethod
    def prepare_documents(
        cls,
        docs: List[Dict[str, Any]],
        chunk_words: int = None,
        overlap_words: int = None,
    ) -> List[Dict[str, Any]]:
        prepared = []
        for doc_index, doc in enumerate(docs):
            text = cls.normalize_text(doc.get("text", ""))
            if not text:
                continue
            base_id = str(doc.get("id") or f"doc-{doc_index}")
            metadata = dict(doc.get("metadata", {}))
            for chunk_index, chunk in enumerate(cls.chunk_text(text, chunk_words, overlap_words)):
                chunk_metadata = {
                    **metadata,
                    "source_doc_id": base_id,
                    "chunk_index": chunk_index,
                }
                prepared.append({
                    "id": f"{base_id}::chunk-{chunk_index}",
                    "text": chunk,
                    "metadata": chunk_metadata,
                })
        return prepared

    @classmethod
    async def index_documents(cls, docs):
        if not cls.initialized:
            logger.warning("RAG service not initialized. Skipping document indexing.")
            return
        prepared_docs = cls.prepare_documents(docs)
        if not prepared_docs:
            logger.warning("No valid documents provided for RAG indexing.")
            return
        embeddings = embeddings_service.embed([doc["text"] for doc in prepared_docs])
        ids = [doc["id"] for doc in prepared_docs]
        documents = [doc["text"] for doc in prepared_docs]
        metadatas = [doc.get("metadata", {}) for doc in prepared_docs]
        chroma_client.add_documents(ids, documents, metadatas, embeddings)
        logger.info("Indexed %d chunks into ChromaDB from %d documents", len(prepared_docs), len(docs))

    @classmethod
    async def retrieve_with_metadata(cls, query: str, top_k: int = 5):
        if not cls.initialized:
            logger.debug("RAG service not initialized. Returning empty results.")
            return []
        query_embeddings = embeddings_service.embed(query)
        results = chroma_client.query(query_embeddings=query_embeddings, n_results=top_k)
        retrieved = []
        documents = results.get("documents", [[]])
        metadatas = results.get("metadatas", [[]])
        distances = results.get("distances", [[]])
        for row_index, row in enumerate(documents):
            metadata_row = metadatas[row_index] if row_index < len(metadatas) else []
            distance_row = distances[row_index] if row_index < len(distances) else []
            for doc_index, document in enumerate(row):
                if not document:
                    continue
                distance = distance_row[doc_index] if doc_index < len(distance_row) else None
                retrieved.append({
                    "text": document,
                    "metadata": metadata_row[doc_index] if doc_index < len(metadata_row) else {},
                    "distance": distance,
                })
        logger.debug("RAG retrieved %d docs", len(retrieved))
        return retrieved

    @classmethod
    async def retrieve(cls, query: str, top_k: int = 5):
        retrieved = await cls.retrieve_with_metadata(query, top_k)
        return [item["text"] for item in retrieved]


rag_service = RAGService()
