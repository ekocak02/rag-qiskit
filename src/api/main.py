import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from src.rag.pipeline import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Qiskit RAG API", version="1.0.0")

# Initialize Pipeline (Singleton pattern to avoid reloading models)
pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    try:
        pipeline = RAGPipeline()
        logger.info("RAG Pipeline loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load RAG Pipeline: {e}")
        raise e

class QueryRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None

class SourceDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]

@app.get("/health")
async def health_check():
    return {"status": "healthy", "pipeline_loaded": pipeline is not None}

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        result = pipeline.run(request.query, request.filters)
        
        # Map result to response model
        sources = [
            SourceDocument(
                content=doc.get('content', ''),
                metadata=doc.get('metadata', {}),
                score=doc.get('rerank_score') or doc.get('score')
            )
            for doc in result.get('source_documents', [])
        ]
        
        return QueryResponse(
            answer=result.get('answer', "No answer generated."),
            sources=sources
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
