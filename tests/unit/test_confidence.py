"""
Unit tests for confidence scoring functionality.
"""
import pytest
from rag.confidence.service import ConfidenceService, ConfidenceResult
from rag.citations.models import CitationResult


class TestConfidenceSignals:
    """Tests for individual confidence signals."""
    
    def setup_method(self):
        self.service = ConfidenceService()
    
    def test_high_retrieval_confidence(self, sample_chunks):
        """High rerank scores should give high confidence."""
        # Modify chunks to have high scores
        for chunk in sample_chunks:
            chunk.score = 8.0
        result = self.service.calculate("Test answer", sample_chunks)
        assert result.breakdown["retrieval"] > 0.7
    
    def test_low_retrieval_confidence(self, sample_chunks):
        """Low rerank scores should give low confidence."""
        for chunk in sample_chunks:
            chunk.score = -3.0
        result = self.service.calculate("Test answer", sample_chunks)
        assert result.breakdown["retrieval"] < 0.3
    
    def test_refusal_lowers_confidence(self, sample_chunks):
        """Refusal should lower confidence."""
        answer = "I cannot answer this question."
        result = self.service.calculate(answer, sample_chunks)
        assert result.breakdown["refusal"] == 0.0
    
    def test_short_answer_lower_confidence(self, sample_chunks):
        """Very short answers should have lower confidence."""
        answer = "Yes."
        result = self.service.calculate(answer, sample_chunks)
        assert result.breakdown["length"] < 0.5


class TestConfidenceAggregation:
    """Tests for confidence score aggregation."""
    
    def test_confidence_levels(self):
        """Test confidence level assignment."""
        low = ConfidenceResult(score=0.3, breakdown={})
        medium = ConfidenceResult(score=0.5, breakdown={})
        high = ConfidenceResult(score=0.8, breakdown={})
        
        assert low.level == "low"
        assert medium.level == "medium"
        assert high.level == "high"
    
    def test_low_confidence_has_disclaimer(self):
        """Low confidence should trigger disclaimer."""
        result = ConfidenceResult(score=0.25, breakdown={})
        assert result.disclaimer is not None
    
    def test_high_confidence_no_disclaimer(self):
        """High confidence should not have disclaimer."""
        result = ConfidenceResult(score=0.9, breakdown={})
        assert result.disclaimer is None
