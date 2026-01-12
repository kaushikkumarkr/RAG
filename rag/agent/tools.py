from typing import List, Dict, Any
from rag.retrieval.service import RetrievalService
from rag.rerank.service import RerankerService

class SearchTool:
    def __init__(self):
        self.retriever = RetrievalService()
        self.reranker = RerankerService()

    def search(self, query: str, top_k: int = 3) -> str:
        """
        Useful for retrieving specific information to answer a question.
        Returns the top-k most relevant text chunks.
        """
        # 1. Hybrid Search
        candidates = self.retriever.hybrid_search(query, top_k=top_k * 2)
        
        # 2. Rerank
        if not candidates:
            return "No information found."
            
        ranked_chunks = self.reranker.rerank(query, candidates, top_k=top_k)
        
        # 3. Format Output
        results = []
        for chunk in ranked_chunks:
            results.append(f"Content: {chunk.content}\nSource: {chunk.metadata}")
            
        return "\n\n".join(results)

    def describe(self) -> str:
        return "SearchTool: Use this to find facts. Input should be a specific search query."
