import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Initialize FastAPI app
app = FastAPI()

# Get the ranker model from the environment variable or default to 'flashrank'
RANKER_MODEL = os.getenv("RANKER_MODEL", "flashrank")

# ColBERT läuft über den CPU-optimierten Pfad (bf16/AMX, Längen-Bucketing, optionaler
# Embedding-Cache). Alle anderen Modelle weiterhin über die rerankers-Bibliothek.
fast_colbert = None
ranker = None
try:
    if RANKER_MODEL == "colbert":
        from rest.fast_colbert import FastColBERT
        fast_colbert = FastColBERT.from_env()
    else:
        from rerankers import Reranker
        ranker = Reranker(RANKER_MODEL)
        if ranker is None:
            raise Exception("Failed to initialize the Reranker.")
except Exception as e:
    print(f"Error initializing the Reranker: {e}")
    raise

# Request model for rerank input
class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    metadata: Optional[List[dict]] = None

# Response model for rerank output
class RerankResponse(BaseModel):
    ranked_documents: List[str]
    scores: List[float]
    metadata: Optional[List[dict]] = None

# Health check route
@app.get("/")
async def read_root():
    info = {"message": f"Rerankers API is running with {RANKER_MODEL}"}
    if fast_colbert is not None and fast_colbert.cache_size > 0:
        info["cache"] = {
            "size": len(fast_colbert._cache),
            "hits": fast_colbert.cache_hits,
            "misses": fast_colbert.cache_misses,
        }
    return info

# Rerank route
@app.post("/rerank", response_model=RerankResponse)
def rerank(request: RerankRequest):
    if len(request.documents) == 0:
        raise HTTPException(status_code=400, detail="The documents list cannot be empty.")
    if request.metadata is not None and len(request.metadata) != len(request.documents):
        raise HTTPException(status_code=400, detail="metadata must have the same length as documents.")

    meta = request.metadata or [{} for _ in request.documents]

    try:
        if fast_colbert is not None:
            scores = fast_colbert.score(request.query, request.documents)
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            ranked_docs = [request.documents[i] for i in order]
            ranked_scores = [scores[i] for i in order]
            ranked_meta = [meta[i] for i in order]
        else:
            from rerankers import Document
            docs = [Document(text=d, metadata=meta[i]) for i, d in enumerate(request.documents)]
            results = ranker.rank(query=request.query, docs=docs)
            ranked_docs = [r.document.text for r in results.results]
            ranked_scores = [r.score for r in results.results]
            ranked_meta = [r.document.metadata for r in results.results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during reranking: {e}")

    return RerankResponse(
        ranked_documents=ranked_docs,
        scores=ranked_scores,
        metadata=ranked_meta if request.metadata else None,
    )
