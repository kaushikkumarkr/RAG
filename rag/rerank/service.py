from typing import List
from opentelemetry import trace
from sentence_transformers import CrossEncoder
from rag.retrieval.models import ScoredChunk

tracer = trace.get_tracer(__name__)

class RerankerService:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # This initializes the model. It might download on first run.
        self.model = CrossEncoder(model_name)

    @tracer.start_as_current_span("rerank")
    def rerank(self, query: str, chunks: List[ScoredChunk], top_k: int = 5) -> List[ScoredChunk]:
        if not chunks:
            return []
            
        # Prepare pairs for cross-encoding
        pairs = [[query, chunk.content] for chunk in chunks]
        
        # CrossEncoder returns scores
        scores = self.model.predict(pairs)
        
        # Update scores and sort
        for i, chunk in enumerate(chunks):
            chunk.score = float(scores[i])
            
        sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
        return sorted_chunks[:top_k]
