from sentence_transformers import SentenceTransformer
import sys

def preload():
    print("Downloading/Loading model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully.")
    
    # Test encoding
    vec = model.encode("Hello world")
    print(f"Test vector dimension: {len(vec)}")

if __name__ == "__main__":
    preload()
