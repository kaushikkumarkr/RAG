import pickle
import os
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from rag.ingestion.models import Chunk

class BM25Index:
    def __init__(self, persistence_path: str = "data/bm25.pkl"):
        self.persistence_path = persistence_path
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Chunk] = []
        self._ensure_data_dir()
        self.load()

    def _ensure_data_dir(self):
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)

    def _tokenize(self, text: str) -> List[str]:
        # Simple tokenization: lowercase and split by whitespace
        # Ideally, use a proper tokenizer (e.g., from nltk or spaCy) matching the language
        return text.lower().split()

    def build(self, chunks: List[Chunk]):
        """
        Builds the BM25 index from a list of chunks.
        Note: This is an in-memory operation and replaces the existing index.
        For production, you'd want incremental updates or a proper search engine (Elastic/OpenSearch).
        """
        self.chunks = chunks
        tokenized_corpus = [self._tokenize(chunk.content) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.save()

    def save(self):
        with open(self.persistence_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)

    def load(self):
        if os.path.exists(self.persistence_path):
            try:
                with open(self.persistence_path, "rb") as f:
                    data = pickle.load(f)
                    self.bm25 = data.get("bm25")
                    self.chunks = data.get("chunks", [])
            except Exception as e:
                print(f"Failed to load BM25 index: {e}")

    def search(self, query: str, top_k: int = 5) -> List[tuple[Chunk, float]]:
        if not self.bm25:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Pair chunks with scores
        chunk_scores = zip(self.chunks, scores)
        
        # Sort by score descending and take top_k
        sorted_results = sorted(chunk_scores, key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
