import os
import json
import logging
import gc
import torch
import chromadb
from typing import List, Dict, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QiskitVectorStore:
    """
    Manages vector database operations for the Qiskit RAG system.
    Handles model loading, embedding generation, and ChromaDB interactions.
    """

    def __init__(self, collection_name: str = "qiskit_rag_collection"):
        self.chroma_path = os.getenv("CHROMA_DB_PATH", "data/vektordb/")
        self.collection_name = collection_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._initialize_db()
        self._load_model()

    def _initialize_db(self):
        """
        Initializes ChromaDB client.
        Detects if running in Docker (Server Mode) or Local (Persistent Mode).
        """
        chroma_host = os.getenv("CHROMA_HOST")
        chroma_port = os.getenv("CHROMA_PORT")

        if chroma_host and chroma_port:
            logger.info(
                f"Connecting to ChromaDB Server at {chroma_host}:{chroma_port}..."
            )
            try:
                self.client = chromadb.HttpClient(
                    host=chroma_host,
                    port=int(chroma_port),
                    settings=chromadb.config.Settings(anonymized_telemetry=False),
                )
                # Trigger a call to ensure connection is valid, though init usually checks
                self.client.heartbeat()
            except Exception as e:
                logger.warning(
                    f"Could not connect to ChromaDB Server at {chroma_host}:{chroma_port} ({e}). Falling back to Local Mode."
                )
                logger.info(f"Running in Local Mode. Database path: {self.chroma_path}")
                self.client = chromadb.PersistentClient(path=self.chroma_path)
        else:
            logger.info(f"Running in Local Mode. Database path: {self.chroma_path}")
            self.client = chromadb.PersistentClient(path=self.chroma_path)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def _load_model(self):
        """Loads the SentenceTransformer model without forcing specific dtypes."""
        logger.info(f"Loading embedding model on {self.device}...")
        model_name = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-300m")
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.model.max_seq_length = 2048
        except Exception as e:
            logger.critical(f"Failed to load model: {e}")
            raise e

    def _format_metadata(
        self, raw_metadata: Dict[str, Any], chunk_id: str
    ) -> Dict[str, Any]:
        """
        Formats metadata for ChromaDB compatibility.
        Converts lists to strings and ensures flat structure.
        """
        formatted = {}
        formatted["chunk_id"] = chunk_id

        if not raw_metadata:
            return formatted

        for key, value in raw_metadata.items():
            if isinstance(value, (str, int, float, bool)):
                formatted[key] = value
            elif isinstance(value, list):
                formatted[key] = ", ".join(map(str, value))

        return formatted

    def _clear_memory(self):
        """Explicitly clears GPU cache to prevent OOM on low-VRAM devices."""
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    def process_and_index(self, jsonl_path: str, batch_size: int = 8):
        """
        Reads the JSONL file, generates embeddings, and indexes data in batches.
        """
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Input file not found: {jsonl_path}")

        logger.info(f"Starting indexing process from: {jsonl_path}")

        docs_buffer, metas_buffer, ids_buffer = [], [], []
        total_indexed = 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Indexing Batches"):
                try:
                    record = json.loads(line)

                    if "page_content" not in record or "chunk_id" not in record:
                        continue

                    content = record["page_content"]
                    chunk_id = record["chunk_id"]
                    metadata = self._format_metadata(
                        record.get("metadata", {}), chunk_id
                    )

                    docs_buffer.append(content)
                    ids_buffer.append(chunk_id)
                    metas_buffer.append(metadata)

                    if len(docs_buffer) >= batch_size:
                        self._embed_and_upsert(docs_buffer, metas_buffer, ids_buffer)
                        total_indexed += len(docs_buffer)

                        docs_buffer, metas_buffer, ids_buffer = [], [], []

                        self._clear_memory()

                except json.JSONDecodeError:
                    logger.warning("Skipped invalid JSON line.")
                    continue

            if docs_buffer:
                self._embed_and_upsert(docs_buffer, metas_buffer, ids_buffer)
                total_indexed += len(docs_buffer)

        logger.info(f"Indexing complete. Total documents: {total_indexed}")

    def _embed_and_upsert(self, docs: List[str], metas: List[Dict], ids: List[str]):
        """Generates embeddings and upserts to ChromaDB."""
        try:
            embeddings = self.model.encode(docs, show_progress_bar=False)

            self.collection.upsert(
                documents=docs, embeddings=embeddings.tolist(), metadatas=metas, ids=ids
            )
        except Exception as e:
            logger.error(f"Failed to upsert batch: {e}")

    def search(
        self, query: str, top_k: int = 20, filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Performs a semantic search on the vector database.

        Args:
            query: The user's query string.
            top_k: Number of results to return.
            filters: Optional metadata filters (e.g., {"has_code": True}).

        Returns:
            Dictionary containing 'documents', 'metadatas', 'distances', 'ids'.
        """
        try:
            query_embedding = self.model.encode(query, convert_to_tensor=False).tolist()

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filters,
                include=["documents", "metadatas", "distances"],
            )
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}
