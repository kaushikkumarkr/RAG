import asyncio
from rag.generation.service import GenerationService
from rag.retrieval.service import RetrievalService
from rag.rerank.service import RerankerService

async def verify_rag():
    print("--- Verifying RAG Pipeline with Document Knowledge ---")
    
    # 1. Ask a question about the Knowledge Base (Paul Graham Essay)
    query = "What is the architecture of the Transformer model?"
    print(f"Query: {query}")
    
    # 2. Retrieval
    retriever = RetrievalService()
    candidates = retriever.hybrid_search(query, top_k=5)
    print(f"Retrieved {len(candidates)} candidates.")
    for i, c in enumerate(candidates):
        print(f"  [{i}] Source: {c.metadata.get('source', 'unknown')} | Content: {c.content[:50]}...")
        
    if not candidates:
        print("❌ Retrieval Failed: No chunks found.")
        return

    # 3. Rerank
    reranker = RerankerService()
    top_chunks = reranker.rerank(query, candidates, top_k=3)
    print(f"Reranked top {len(top_chunks)} chunks.")
    
    # 4. Generate
    generator = GenerationService()
    answer = generator.generate_answer(query, top_chunks)
    
    print("\nGenerated Answer:")
    print("--------------------------------------------------")
    print(answer)
    print("--------------------------------------------------")
    
    if "cannot answer" in answer.lower():
        print("⚠️  Result: LLM declined to answer (Context might be missing specifics).")
    else:
        print("✅ Result: Success! Valid answer generated from context.")

if __name__ == "__main__":
    asyncio.run(verify_rag())
