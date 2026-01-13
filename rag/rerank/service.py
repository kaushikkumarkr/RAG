from typing import List, Optional
from langfuse import Langfuse
from sentence_transformers import CrossEncoder
from rag.retrieval.models import ScoredChunk

# Initialize Langfuse for manual tracing
langfuse = Langfuse()

class RerankerService:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # This initializes the model. It might download on first run.
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: List[ScoredChunk], top_k: int = 5, observation=None) -> List[ScoredChunk]:
        """
        Rerank retrieved chunks using a cross-encoder model.
        
        Args:
            query: The user's question.
            chunks: Retrieved context chunks to rerank.
            top_k: Number of top chunks to return.
            observation: Optional Langfuse observation (trace/span) to nest under.
                        If provided, creates a child span. Otherwise, creates standalone trace.
        """
        if not chunks:
            return []
        
        # Create span (nested or standalone)
        if observation:
            span = observation.span(
                name="rerank",
                input={"query": query, "num_chunks": len(chunks), "top_k": top_k}
            )
        else:
            span = langfuse.trace(
                name="rerank",
                input={"query": query, "num_chunks": len(chunks), "top_k": top_k}
            )
        
        try:
            # Prepare pairs for cross-encoding
            pairs = [[query, chunk.content] for chunk in chunks]
            
            # CrossEncoder returns scores
            scores = self.model.predict(pairs)
            
            # Update scores and sort
            for i, chunk in enumerate(chunks):
                chunk.score = float(scores[i])
                
            sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
            result = sorted_chunks[:top_k]
            
            span.end(output={"num_results": len(result), "top_score": result[0].score if result else 0})
            return result
        except Exception as e:
            span.end(output={"error": str(e)})
            raise
