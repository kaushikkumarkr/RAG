"""
Unified Citation Service: Extract, validate, and format citations.
"""
from typing import List, Optional
from langfuse import Langfuse
from rag.citations.models import Citation, CitationResult, SourceReference
from rag.citations.extractor import CitationExtractor
from rag.citations.validator import CitationValidator
from rag.citations.formatter import CitationFormatter
from rag.retrieval.models import ScoredChunk

langfuse = Langfuse()


class CitationService:
    """
    Unified service for handling citations in RAG answers.
    
    Usage:
        service = CitationService()
        result = service.process(answer, chunks)
        
        print(result.formatted_answer)  # Answer with [1], [2], etc.
        print(result.sources)           # List of SourceReference
        print(result.phantom_citations) # Hallucinated refs (if any)
    """
    
    def __init__(self):
        self.extractor = CitationExtractor()
        self.validator = CitationValidator()
        self.formatter = CitationFormatter()
    
    def process(
        self, 
        answer: str, 
        chunks: List[ScoredChunk],
        observation=None
    ) -> CitationResult:
        """
        Process an answer: extract, validate, and format citations.
        
        Args:
            answer: LLM-generated answer with [doc_id:chunk_index] citations.
            chunks: Retrieved chunks for validation.
            observation: Optional Langfuse observation for logging.
        
        Returns:
            CitationResult with formatted answer and source references.
        """
        span = None
        if observation:
            span = observation.span(
                name="citation_processing",
                input={"answer_length": len(answer), "num_chunks": len(chunks)}
            )
        
        # 1. Extract citations
        citations = self.extractor.extract(answer)
        
        # 2. Validate citations
        valid_citations, phantom_citations = self.validator.validate(citations, chunks)
        
        # 3. Format with numbered references
        formatted_answer, sources = self.formatter.to_numbered(
            answer, valid_citations, chunks
        )
        
        result = CitationResult(
            citations=citations,
            valid_citations=valid_citations,
            phantom_citations=phantom_citations,
            sources=sources,
            formatted_answer=formatted_answer
        )
        
        if span:
            span.end(output={
                "total_citations": len(citations),
                "valid_citations": len(valid_citations),
                "phantom_citations": len(phantom_citations),
                "sources": len(sources)
            })
        
        return result
    
    def get_sources_markdown(self, result: CitationResult) -> str:
        """Generate markdown sources section."""
        return self.formatter.generate_sources_section(result.sources)
    
    def get_answer_with_sources(self, result: CitationResult) -> str:
        """Get complete answer with sources section appended."""
        sources_section = self.get_sources_markdown(result)
        return f"{result.formatted_answer}\n{sources_section}"
