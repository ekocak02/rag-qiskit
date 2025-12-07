import logging
from typing import Dict, Any
from src.rag.retriever import QiskitRetriever
from src.rag.reranker import CrossEncoderReranker
from src.rag.generator import GeminiGenerator

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orchestrates the RAG flow: Retrieve -> Rerank -> Generate.
    """

    def __init__(self):
        logger.info("Initializing RAG Pipeline...")
        self.retriever = QiskitRetriever()
        self.reranker = CrossEncoderReranker()
        self.generator = GeminiGenerator()
        logger.info("RAG Pipeline Initialized.")

    def run(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        End-to-end RAG execution.
        """
        # 1. Retrieve
        retrieved_docs = self.retriever.retrieve(query, top_k=50, filters=filters)
        logger.info(f"Retrieved {len(retrieved_docs)} documents.")

        if not retrieved_docs:
            return {"answer": "No relevant documents found.", "source_documents": []}

        # 2. Rerank
        reranked_docs = self.reranker.rerank(query, retrieved_docs, top_n=5)
        logger.info(f"Top {len(reranked_docs)} documents selected after reranking.")

        # 3. Generate
        answer = self.generator.generate_answer(query, reranked_docs)

        return {"answer": answer, "source_documents": reranked_docs}
