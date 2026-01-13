from typing import List, Dict, Any
import requests
try:
    from langfuse.decorators import observe
except ImportError:
    from langfuse import observe
from rag.retrieval.service import RetrievalService
from rag.rerank.service import RerankerService

class SearchTool:
    def __init__(self):
        self.retriever = RetrievalService()
        self.reranker = RerankerService()

    @observe(as_type="generation")
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

class CryptoPriceTool:
    def __init__(self):
        self.base_url = "https://api.coincap.io/v2/assets"

    @observe(as_type="generation")
    def get_price(self, symbol: str) -> str:
        """
        Useful for getting the LIVE price of a cryptocurrency.
        Input should be the symbol name (e.g., 'bitcoin', 'ethereum').
        """
        try:
            # Clean input
            symbol = symbol.lower().strip().replace('"', '')
            
            # Direct mapping for common aliases if needed, but CoinCap uses id (bitcoin, ethereum)
            if symbol == "btc": symbol = "bitcoin"
            if symbol == "eth": symbol = "ethereum"
            
            url = f"{self.base_url}/{symbol}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                price = float(data['data']['priceUsd'])
                return f"The current price of {symbol} is ${price:,.2f} USD."
            else:
                return f"Error: Could not fetch price for {symbol} (Status {response.status_code})."
        except Exception as e:
            return f"Error fetching price: {str(e)}"

    def describe(self) -> str:
        return "CryptoPriceTool: Use this to get live cryptocurrency prices. Input should be the full name (e.g., bitcoin)."
