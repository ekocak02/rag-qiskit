import logging
from typing import List, Dict, Any
from src.database.storage_manager import QiskitVectorStore

logger = logging.getLogger(__name__)


class QiskitRetriever:
    """
    Retrieves relevant documents from the vector store.
    """

    def __init__(self, vector_store: QiskitVectorStore = None):
        """
        Args:
            vector_store: Existing QiskitVectorStore instance. If None, creates a new one.
        """
        self.store = vector_store if vector_store else QiskitVectorStore()

    def retrieve(
        self, query: str, top_k: int = 50, filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k documents matching the query.

        Returns:
            List of dictionaries with 'content', 'metadata', 'score'.
        """
        logger.info(f"Retrieving top {top_k} for query: {query}")

        results = self.store.search(query, top_k=top_k, filters=filters)

        # Parse ChromaDB results into a cleaner format
        parsed_results = []
        if not results["documents"]:
            return []

        # Chroma returns list of lists (batch query support), we sent 1 query
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        for i in range(len(docs)):
            parsed_results.append(
                {
                    "id": ids[i],
                    "content": docs[i],
                    "metadata": metas[i],
                    "score": distances[i],
                }
            )

        return parsed_results
