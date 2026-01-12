from typing import List
from opentelemetry import trace
from rag.retrieval.models import ScoredChunk
from rag.embeddings.service import EmbeddingService
from rag.vector_store.qdrant import QdrantService
from rag.sparse.index import BM25Index
from apps.api.settings import settings

tracer = trace.get_tracer(__name__)

class RetrievalService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService(url=settings.QDRANT_URL)
        self.bm25_index = BM25Index()

    @tracer.start_as_current_span("search")
    def search(self, query: str, top_k: int = 5) -> List[ScoredChunk]:
        # 1. Embed query
        query_vector = self.embedding_service.embed_query(query)
        
        # 2. Search Qdrant
        results = self.qdrant_service.search(query_vector=query_vector, limit=top_k)
        
        # 3. Format results
        scored_chunks = []
        for point in results:
            payload = point.payload or {}
            scored_chunks.append(ScoredChunk(
                content=payload.get("content", ""),
                score=point.score,
                doc_id=payload.get("doc_id", ""),
                chunk_index=payload.get("chunk_index") or -1,
                metadata=payload
            ))
            
        return scored_chunks

    @tracer.start_as_current_span("hybrid_search")
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[ScoredChunk]:
        """
        Alpha: Weight for dense search (0.0 to 1.0).
        Score = alpha * dense + (1 - alpha) * sparse
        Note: This assumes raw scores are comparable or requires normalization.
        For MVP, we'll try raw normalization on the fly.
        """
        # 1. Get Dense Results
        # We fetch more than top_k to increase overlap potential
        dense_results = self.qdrant_service.search(
            query_vector=self.embedding_service.embed_query(query),
            limit=top_k * 2
        )
        
        # 2. Get Sparse Results
        sparse_results = self.bm25_index.search(query, top_k=top_k * 2)
        
        # 3. Normalize Scores
        # Simple Min-Max Normalization
        dense_scores = [p.score for p in dense_results]
        sparse_scores = [s for _, s in sparse_results]
        
        max_d = max(dense_scores) if dense_scores else 1.0
        min_d = min(dense_scores) if dense_scores else 0.0
        
        max_s = max(sparse_scores) if sparse_scores else 1.0
        min_s = min(sparse_scores) if sparse_scores else 0.0
        
        def norm_d(s): return (s - min_d) / (max_d - min_d + 1e-6)
        def norm_s(s): return (s - min_s) / (max_s - min_s + 1e-6)
        
        # 4. Merge
        # We need a map of content -> combined_score to deduplicate
        combined = {}
        
        for point in dense_results:
            content = point.payload.get("content")
            score = norm_d(point.score)
            combined[content] = {
                "score": score * alpha,
                "chunk": point,
                "type": "dense",
                "original_chunk_obj": None
            }
            
        for chunk, score in sparse_results:
            n_score = norm_s(score)
            weighted = n_score * (1 - alpha)
            if chunk.content in combined:
                combined[chunk.content]["score"] += weighted
            else:
                combined[chunk.content] = {
                    "score": weighted,
                    "chunk": None,
                    "type": "sparse",
                    "original_chunk_obj": chunk
                }
                
        # 5. Sort & Format
        sorted_items = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        final_results = []
        
        for item in sorted_items[:top_k]:
            if item["type"] == "dense":
                 payload = item["chunk"].payload
                 
                 final_results.append(ScoredChunk(
                     content=payload.get("content"),
                     score=item["score"],
                     doc_id=payload.get("doc_id", ""),
                     chunk_index=payload.get("chunk_index") if payload.get("chunk_index") is not None else -1,
                     metadata=payload
                 ))
            else:
                 chunk = item["original_chunk_obj"]
                 final_results.append(ScoredChunk(
                     content=chunk.content,
                     score=item["score"],
                     doc_id=chunk.doc_id,
                     chunk_index=chunk.chunk_index,
                     metadata=chunk.metadata
                 ))
                 
        return final_results
