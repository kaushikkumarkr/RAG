from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from rag.ingestion.models import Chunk
import os

class QdrantService:
    def __init__(self, url: str, collection_name: str = "rag_foundry_dense", vector_size: int = 384):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )

    def upsert_chunks(self, chunks: List[Chunk]):
        if not chunks:
            return
            
        points = [
            models.PointStruct(
                id=chunk.id,
                vector=chunk.vector,
                payload={
                    "content": chunk.content,
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata
                }
            )
            for chunk in chunks
            if chunk.vector is not None
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_vector: List[float], limit: int = 5) -> List[models.ScoredPoint]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        )
        return response.points
