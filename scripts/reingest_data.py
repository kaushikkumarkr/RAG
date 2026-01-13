import asyncio
from rag.ingestion.service import IngestionService
from rag.ingestion.models import Document
import os

async def ingest_samples():
    print("--- Re-Ingesting Sample Data ---")
    service = IngestionService()
    
    dir_path = "sample_data/large_dataset"
    if os.path.exists(dir_path):
        chunks = service.ingest_directory(dir_path)
        print(f"✅ Ingested {chunks} chunks from directory: {dir_path}")
    else:
        print(f"❌ Directory not found: {dir_path}")

if __name__ == "__main__":
    asyncio.run(ingest_samples())
