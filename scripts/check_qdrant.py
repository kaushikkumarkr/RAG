from qdrant_client import QdrantClient
from apps.api.settings import settings

def check_stats():
    print(f"Connecting to Qdrant at {settings.QDRANT_URL}...")
    client = QdrantClient(url=settings.QDRANT_URL)
    
    collection_name = "rag_foundry_dense"
    
    try:
        info = client.get_collection(collection_name)
        print(f"\n✅ Collection '{collection_name}' exists.")
        print(f"📊 Total Vectors (Points): {info.points_count}")
        print(f"🔹 Status: {info.status}")
    except Exception as e:
        print(f"\n❌ Error accessing collection '{collection_name}': {e}")

if __name__ == "__main__":
    check_stats()
