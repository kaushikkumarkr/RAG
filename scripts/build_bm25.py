from rag.ingestion.loaders import LoaderFactory
from rag.chunking.splitter import RecursiveSplitter
from rag.sparse.index import BM25Index
from rag.ingestion.models import Chunk
import os

def build_bm25_only():
    print("Building BM25 index from scratch...")
    
    file_path = "sample_data/paul_graham_lisp.txt"
    loader = LoaderFactory.get_loader(file_path)
    documents = loader.load(file_path)
    
    splitter = RecursiveSplitter()
    chunks = []
    
    for doc in documents:
        chunks.extend(splitter.split(doc))
        
    print(f"Generated {len(chunks)} chunks.")
    
    index = BM25Index()
    index.build(chunks)
    print("BM25 index built and saved to data/bm25.pkl")

if __name__ == "__main__":
    build_bm25_only()
