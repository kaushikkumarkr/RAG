import sys
print("DEBUG: Starting detailed import check...")

print("DEBUG: Importing models...")
from rag.ingestion.models import Document
print("DEBUG: Importing loaders...")
from rag.ingestion.loaders import LoaderFactory
print("DEBUG: Importing splitter...")
from rag.chunking.splitter import RecursiveSplitter
print("DEBUG: Importing embedding service...")
from rag.embeddings.service import EmbeddingService
print("DEBUG: Importing qdrant service...")
from rag.vector_store.qdrant import QdrantService
print("DEBUG: Importing ingestion service...")
from rag.ingestion.service import IngestionService

print("DEBUG: All imports successful.")

def debug():
    print("DEBUG: Initializing IngestionService...")
    service = IngestionService()
    print("DEBUG: IngestionService initialized.")
    
    file_path = "sample_data/paul_graham_lisp.txt"
    print(f"DEBUG: Ingesting {file_path}...")
    
    try:
        count = service.ingest_file(file_path)
        print(f"DEBUG: Successfully ingested {count} chunks.")
        
        import os
        if os.path.exists("data/bm25.pkl"):
             print("DEBUG: BM25 pickle created successfully.")
        else:
             print("DEBUG: BM25 pickle NOT found.")
             
    except Exception as e:
        print(f"DEBUG: Error during ingestion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug()
