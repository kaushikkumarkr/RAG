"""
Unit tests for citations functionality.
"""
import pytest
from rag.citations.service import CitationService
from rag.retrieval.models import ScoredChunk


class TestCitationExtractor:
    """Tests for citation extraction."""
    
    def setup_method(self):
        self.service = CitationService()
    
    def test_extract_single_citation(self, sample_chunks):
        """Single citation should be extracted."""
        text = "The Transformer uses attention [test-doc-1:0]."
        result = self.service.process(text, sample_chunks)
        assert result.citation_count == 1
    
    def test_extract_multiple_citations(self, sample_chunks):
        """Multiple citations should be extracted."""
        text = "First [test-doc-1:0] and second [test-doc-1:1]."
        result = self.service.process(text, sample_chunks)
        assert result.citation_count == 2
    
    def test_no_citations(self, sample_chunks):
        """Text without citations should return empty list."""
        text = "The Transformer uses attention."
        result = self.service.process(text, sample_chunks)
        assert result.citation_count == 0


class TestCitationValidator:
    """Tests for citation validation."""
    
    def setup_method(self):
        self.service = CitationService()
    
    def test_valid_citations(self, sample_chunks):
        """Valid citations should be recognized."""
        text = "The Transformer uses attention [test-doc-1:0]."
        result = self.service.process(text, sample_chunks)
        assert result.valid_count == 1
        assert not result.has_phantoms
    
    def test_phantom_citation_detected(self, sample_chunks):
        """Phantom citations should be flagged."""
        text = "Uses RNNs [fake-doc:99]."
        result = self.service.process(text, sample_chunks)
        assert result.has_phantoms


class TestCitationFormatter:
    """Tests for citation formatting."""
    
    def setup_method(self):
        self.service = CitationService()
    
    def test_format_to_numbered(self, sample_chunks):
        """Citations should be converted to numbered format."""
        text = "Uses attention [test-doc-1:0]."
        result = self.service.process(text, sample_chunks)
        assert "[1]" in result.formatted_answer
    
    def test_duplicate_citations_same_number(self, sample_chunks):
        """Duplicate citations should get same number."""
        text = "First [test-doc-1:0] and again [test-doc-1:0]."
        result = self.service.process(text, sample_chunks)
        assert len(result.sources) == 1  # Only one unique source
