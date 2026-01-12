from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
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

class AskResponse(BaseModel):
    answer: str
    citations: List[ScoredChunk]

@router.post("", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        # 1. Retrieval
        retrieval_service = RetrievalService()
        if request.use_hybrid:
            # Fetch more candidates for reranking
            candidates = retrieval_service.hybrid_search(request.question, top_k=20) 
        else:
            candidates = retrieval_service.search(request.question, top_k=20)
            
        if not candidates:
            return AskResponse(answer="I found no relevant information in the knowledge base.", citations=[])

        # 2. Reranking
        reranker_service = RerankerService()
        top_chunks = reranker_service.rerank(request.question, candidates, top_k=5)
        
        # 3. Generation
        generation_service = GenerationService()
        answer = generation_service.generate_answer(request.question, top_chunks)
        
        return AskResponse(answer=answer, citations=top_chunks)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
