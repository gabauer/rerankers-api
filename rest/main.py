import os
import subprocess
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rerankers import Reranker, Document
from typing import List, Optional
from rerankers.results import RankedResults

# Initialize FastAPI app
app = FastAPI()

# Get the ranker model from the environment variable or default to 'flashrank'
RANKER_MODEL = os.getenv("RANKER_MODEL", "flashrank")

# Map ranker models to pip dependencies
dependencies = {
    "colbert": "rerankers[transformers]",
    "flashrank": "rerankers[flashrank]",
}

# Install the necessary dependencies
dependency = dependencies.get(RANKER_MODEL)
if dependency:
    result = subprocess.run([sys.executable, "-m", "pip", "install", dependency], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to install {dependency}: {result.stderr}")
        raise Exception(f"Dependency installation failed: {result.stderr}")
    else:
        print(f"Successfully installed {dependency}: {result.stdout}")
else:
    raise Exception(f"Unknown ranker model: {RANKER_MODEL}")

# Initialize the Reranker based on the environment
try:
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
    return {"message": f"Rerankers API is running with {RANKER_MODEL}"}

# Rerank route
@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    if len(request.documents) == 0:
        raise HTTPException(status_code=400, detail="The documents list cannot be empty.")

    # Prepare documents for reranking
    docs = [Document(text=doc, metadata=(request.metadata[i] if request.metadata else {})) for i, doc in enumerate(request.documents)]

    # Perform reranking using the specified query and documents
    try:
        results: RankedResults = ranker.rank(query=request.query, docs=docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during reranking: {e}")

    ranked_docs = [result.document.text for result in results.results]
    scores = [result.score for result in results.results]
    metadata = [result.document.metadata for result in results.results]

    return RerankResponse(ranked_documents=ranked_docs, scores=scores, metadata=metadata if request.metadata else None)
