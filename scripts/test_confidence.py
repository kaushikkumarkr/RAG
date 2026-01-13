"""
Test script for confidence scoring functionality.
"""
from rag.confidence.service import ConfidenceService
from rag.citations.service import CitationService
from rag.retrieval.models import ScoredChunk

def test_confidence():
    print("=" * 70)
    print("📊 CONFIDENCE SCORING TEST SUITE")
    print("=" * 70)
    
    confidence_service = ConfidenceService()
    citation_service = CitationService()
    
    # Mock high-quality chunks
    good_chunks = [
        ScoredChunk(
            content="The Transformer model uses self-attention mechanisms.",
            score=7.5,  # High rerank score
            doc_id="abc123",
            chunk_index=0,
            metadata={"source": "transformer.pdf"}
        ),
        ScoredChunk(
            content="Self-attention allows the model to focus on relevant parts.",
            score=6.8,
            doc_id="abc123",
            chunk_index=1,
            metadata={"source": "transformer.pdf"}
        ),
        ScoredChunk(
            content="Attention mechanisms are key to Transformer performance.",
            score=5.5,
            doc_id="abc123",
            chunk_index=2,
            metadata={"source": "transformer.pdf"}
        ),
    ]
    
    # Mock low-quality chunks
    bad_chunks = [
        ScoredChunk(
            content="Unrelated content about cooking.",
            score=-2.0,
            doc_id="xyz",
            chunk_index=0,
            metadata={}
        ),
    ]
    
    # Test 1: High confidence answer
    print("\n" + "=" * 60)
    print("📋 Test 1: High Confidence Answer")
    print("=" * 60)
    
    answer1 = (
        "The Transformer model uses self-attention mechanisms [abc123:0] "
        "to process sequences. This attention [abc123:1] allows the model "
        "to focus on relevant parts of the input [abc123:2]."
    )
    
    citation_result1 = citation_service.process(answer1, good_chunks)
    result1 = confidence_service.calculate(answer1, good_chunks, citation_result1)
    
    print(f"Answer: {answer1[:60]}...")
    print(f"Confidence: {result1}")
    print(f"Breakdown: {result1.breakdown}")
    print(f"Disclaimer: {result1.disclaimer or 'None'}")
    
    # Test 2: Low confidence (refusal)
    print("\n" + "=" * 60)
    print("📋 Test 2: Low Confidence (Refusal)")
    print("=" * 60)
    
    answer2 = "I cannot answer this question based on the provided information."
    
    citation_result2 = citation_service.process(answer2, bad_chunks)
    result2 = confidence_service.calculate(answer2, bad_chunks, citation_result2)
    
    print(f"Answer: {answer2}")
    print(f"Confidence: {result2}")
    print(f"Breakdown: {result2.breakdown}")
    print(f"Disclaimer: {result2.disclaimer or 'None'}")
    
    # Test 3: Medium confidence (short answer, no citations)
    print("\n" + "=" * 60)
    print("📋 Test 3: Medium Confidence (Short, No Citations)")
    print("=" * 60)
    
    answer3 = "Transformers use attention."
    
    citation_result3 = citation_service.process(answer3, good_chunks)
    result3 = confidence_service.calculate(answer3, good_chunks, citation_result3)
    
    print(f"Answer: {answer3}")
    print(f"Confidence: {result3}")
    print(f"Breakdown: {result3.breakdown}")
    print(f"Disclaimer: {result3.disclaimer or 'None'}")
    
    # Test 4: Mixed confidence
    print("\n" + "=" * 60)
    print("📋 Test 4: Check Disclaimer Logic")
    print("=" * 60)
    
    # Create answers at different confidence levels
    levels = [0.2, 0.4, 0.6, 0.8]
    for level in levels:
        # Manually create result to test disclaimer
        from rag.confidence.service import ConfidenceResult
        test_result = ConfidenceResult(score=level, breakdown={})
        print(f"Score: {level:.1f} → Level: {test_result.level}, Disclaimer: {bool(test_result.disclaimer)}")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_confidence()
