from typing import List
import os
from rag.ingestion.loaders import LoaderFactory
from rag.chunking.splitter import RecursiveSplitter
from rag.embeddings.service import EmbeddingService
from rag.vector_store.qdrant import QdrantService
from rag.sparse.index import BM25Index
from rag.ingestion.models import Chunk
from apps.api.settings import settings

class IngestionService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService(url=settings.QDRANT_URL)
        self.splitter = RecursiveSplitter()
        self.bm25_index = BM25Index()

    def ingest_file(self, file_path: str) -> int:
        """
        Ingests a single file: Load -> Chunk -> Embed -> Index
        Returns the number of chunks indexed.
        """
        # 1. Load
        loader = LoaderFactory.get_loader(file_path)
        documents = loader.load(file_path)
        
        all_chunks: List[Chunk] = []
        
        # 2. Chunk
        for doc in documents:
            chunks = self.splitter.split(doc)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            return 0
            
        # 3. Embed
        texts = [chunk.content for chunk in all_chunks]
        vectors = self.embedding_service.embed(texts)
        
        for i, chunk in enumerate(all_chunks):
            chunk.vector = vectors[i]
            
        # 4. Index Dense
        self.qdrant_service.upsert_chunks(all_chunks)

        # 5. Index Sparse (Append and Rebuild)
        # In a real app, this might be async or batch
        current_chunks = self.bm25_index.chunks
        current_chunks.extend(all_chunks)
        self.bm25_index.build(current_chunks)
        
        return len(all_chunks)

    def ingest_directory(self, directory_path: str) -> int:
        """
        Ingests all supported files in a directory.
        """
        total_chunks = 0
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(('.txt', '.md', '.pdf')):
                    file_path = os.path.join(root, file)
                    try:
                        print(f"Ingesting {file_path}...")
                        count = self.ingest_file(file_path)
                        total_chunks += count
                    except Exception as e:
                        print(f"Failed to ingest {file_path}: {e}")
        return total_chunks
