"""
Test script for citations functionality.
"""
from rag.citations.service import CitationService
from rag.retrieval.models import ScoredChunk

def test_citations():
    print("=" * 70)
    print("📚 CITATIONS TEST SUITE")
    print("=" * 70)
    
    service = CitationService()
    
    # Mock chunks
    mock_chunks = [
        ScoredChunk(
            content="The Transformer model uses self-attention mechanisms to process sequences.",
            score=0.9,
            doc_id="abc123-def456",
            chunk_index=0,
            metadata={"source": "attention_paper.pdf"}
        ),
        ScoredChunk(
            content="Attention allows the model to focus on relevant parts of the input.",
            score=0.85,
            doc_id="abc123-def456",
            chunk_index=1,
            metadata={"source": "attention_paper.pdf"}
        ),
        ScoredChunk(
            content="GPT-4 is a large language model trained on diverse data.",
            score=0.7,
            doc_id="xyz789-uvw123",
            chunk_index=5,
            metadata={"source": "gpt4_report.pdf"}
        ),
    ]
    
    # Test 1: Answer with valid citations
    print("\n" + "=" * 60)
    print("📋 Test 1: Valid Citations")
    print("=" * 60)
    
    answer1 = (
        "The Transformer uses attention [abc123-def456:0] to handle sequences. "
        "This attention mechanism [abc123-def456:1] helps focus on relevant input. "
        "GPT-4 [xyz789-uvw123:5] builds on these ideas."
    )
    
    result1 = service.process(answer1, mock_chunks)
    print(f"Original: {answer1[:80]}...")
    print(f"Formatted: {result1.formatted_answer[:80]}...")
    print(f"Summary: {result1.summary()}")
    print(f"Sources: {len(result1.sources)}")
    print("\n" + service.get_sources_markdown(result1))
    
    # Test 2: Answer with phantom (hallucinated) citation
    print("\n" + "=" * 60)
    print("📋 Test 2: Phantom Citation (Hallucination)")
    print("=" * 60)
    
    answer2 = (
        "The model uses RNNs [fake-doc-id:99] for processing. "
        "But Transformers [abc123-def456:0] are better."
    )
    
    result2 = service.process(answer2, mock_chunks)
    print(f"Original: {answer2}")
    print(f"Formatted: {result2.formatted_answer}")
    print(f"Summary: {result2.summary()}")
    print(f"Phantom citations: {result2.phantom_citations}")
    
    # Test 3: Answer without citations
    print("\n" + "=" * 60)
    print("📋 Test 3: No Citations")
    print("=" * 60)
    
    answer3 = "The Transformer model is based on attention mechanisms."
    
    result3 = service.process(answer3, mock_chunks)
    print(f"Original: {answer3}")
    print(f"Formatted: {result3.formatted_answer}")
    print(f"Citation count: {result3.citation_count}")
    print(f"Summary: {result3.summary()}")
    
    # Test 4: Duplicate citations
    print("\n" + "=" * 60)
    print("📋 Test 4: Duplicate Citations (Same Source)")
    print("=" * 60)
    
    answer4 = (
        "Attention [abc123-def456:0] is key. "
        "As mentioned [abc123-def456:0], it helps focus. "
        "This attention [abc123-def456:0] transforms NLP."
    )
    
    result4 = service.process(answer4, mock_chunks)
    print(f"Original: {answer4}")
    print(f"Formatted: {result4.formatted_answer}")
    print(f"Citations extracted: {result4.citation_count}")
    print(f"Unique sources: {len(result4.sources)}")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_citations()
