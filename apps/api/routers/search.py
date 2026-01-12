from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from rag.retrieval.service import RetrievalService
from rag.retrieval.models import ScoredChunk

router = APIRouter(prefix="/search", tags=["Search"])

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResponse(BaseModel):
    results: List[ScoredChunk]

@router.post("/dense", response_model=SearchResponse)
async def search_dense(request: SearchRequest):
    service = RetrievalService()
    try:
        results = service.search(query=request.query, top_k=request.top_k)
        return SearchResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class HybridSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    alpha: float = 0.5

@router.post("/hybrid", response_model=SearchResponse)
async def search_hybrid(request: HybridSearchRequest):
    service = RetrievalService()
    try:
        results = service.hybrid_search(
            query=request.query, 
            top_k=request.top_k, 
            alpha=request.alpha
        )
        return SearchResponse(results=results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
