from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from rag.retrieval.service import RetrievalService
from rag.rerank.service import RerankerService
from rag.generation.service import GenerationService
from rag.retrieval.models import ScoredChunk

router = APIRouter(prefix="/ask", tags=["Ask"])

class AskRequest(BaseModel):
    question: str
    filters: Optional[Dict[str, Any]] = None
    use_hybrid: bool = True
    top_k: int = Field(default=5, ge=1, le=20)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)

class AskResponse(BaseModel):
    answer: str
    citations: List[ScoredChunk]

@router.post("", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        # 1. Retrieval
        retrieval_service = RetrievalService()
        candidate_k = min(request.top_k * 4, 80)
        if request.use_hybrid:
            # Fetch more candidates for reranking
            candidates = retrieval_service.hybrid_search(
                request.question, top_k=candidate_k, alpha=request.alpha
            )
        else:
            candidates = retrieval_service.search(request.question, top_k=candidate_k)
            
        if not candidates:
            return AskResponse(answer="I found no relevant information in the knowledge base.", citations=[])

        # 2. Reranking
        reranker_service = RerankerService()
        top_chunks = reranker_service.rerank(
            request.question, candidates, top_k=request.top_k
        )
        
        # 3. Generation
        generation_service = GenerationService()
        answer = generation_service.generate_answer(request.question, top_chunks)
        
        return AskResponse(answer=answer, citations=top_chunks)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
