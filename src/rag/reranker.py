import logging
import torch
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Re-ranks retrieved documents using a Cross-Encoder model.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Reranker model {model_name} on {self.device}...")
        try:
            self.model = CrossEncoder(model_name, device=self.device)
        except Exception as e:
            logger.critical(f"Failed to load Reranker model: {e}")
            raise e

    def rerank(
        self, query: str, documents: List[Dict[str, Any]], top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Reranks the documents based on relevance to the query.

        Args:
            query: User query.
            documents: List of dicts from Retriever (must have 'content').
            top_n: Number of documents to return after reranking.

        Returns:
            Top-N documents with updated 'score' (relevance score).
        """
        if not documents:
            return []

        # Prepare pairs for Cross-Encoder
        pairs = [[query, doc["content"]] for doc in documents]

        # Predict scores
        scores = self.model.predict(pairs)

        # Attach scores to documents
        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])

        # Sort by score (descending)
        sorted_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

        return sorted_docs[:top_n]
