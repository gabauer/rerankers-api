import os
import torch
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rerankers import Reranker, Document
from typing import List, Optional
from rerankers.results import RankedResults

torch.set_num_threads(2)

app = FastAPI()

RANKER_MODEL = os.getenv("RANKER_MODEL", "flashrank")

model_lock = asyncio.Lock()

try:
    ranker = Reranker(RANKER_MODEL, device="cpu")
    if ranker is None:
        raise Exception("Failed to initialize the Reranker.")
except Exception as e:
    print(f"Error initializing the Reranker: {e}")
    raise

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    metadata: Optional[List[dict]] = None

class RerankResponse(BaseModel):
    ranked_documents: List[str]
    scores: List[float]
    metadata: Optional[List[dict]] = None

@app.get("/")
async def read_root():
    return {"message": f"Rerankers API is running with {RANKER_MODEL}"}

@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    if len(request.documents) == 0:
        raise HTTPException(status_code=400, detail="The documents list cannot be empty.")

    docs = [Document(text=doc, metadata=(request.metadata[i] if request.metadata else {}))
            for i, doc in enumerate(request.documents)]

    try:
        async with model_lock:
            results: RankedResults = await asyncio.to_thread(
                ranker.rank, query=request.query, docs=docs
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during reranking: {e}")

    ranked_docs = [result.document.text for result in results.results]
    scores = [result.score for result in results.results]
    metadata = [result.document.metadata for result in results.results]

    return RerankResponse(
        ranked_documents=ranked_docs,
        scores=scores,
        metadata=metadata if request.metadata else None
    )