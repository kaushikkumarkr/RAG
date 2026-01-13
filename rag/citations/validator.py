"""
Citation Validator: Verify citations exist in retrieved chunks.
"""
from typing import List, Set, Tuple
from rag.citations.models import Citation
from rag.retrieval.models import ScoredChunk


class CitationValidator:
    """
    Validate that citations reference actual retrieved chunks.
    
    Detects "phantom citations" - hallucinated references to sources
    that don't exist in the retrieval context.
    """
    
    def validate(
        self, 
        citations: List[Citation], 
        chunks: List[ScoredChunk]
    ) -> Tuple[List[Citation], List[Citation]]:
        """
        Validate citations against retrieved chunks.
        
        Args:
            citations: List of citations extracted from answer.
            chunks: List of retrieved chunks.
        
        Returns:
            Tuple of (valid_citations, phantom_citations)
        """
        # Build set of valid citation keys from chunks
        valid_keys: Set[str] = set()
        for chunk in chunks:
            key = f"{chunk.doc_id}:{chunk.chunk_index}"
            valid_keys.add(key)
        
        valid_citations = []
        phantom_citations = []
        
        for citation in citations:
            if citation.key in valid_keys:
                valid_citations.append(citation)
            else:
                phantom_citations.append(citation)
        
        return valid_citations, phantom_citations
    
    def get_chunk_for_citation(
        self, 
        citation: Citation, 
        chunks: List[ScoredChunk]
    ) -> ScoredChunk | None:
        """Get the chunk that a citation references."""
        for chunk in chunks:
            if chunk.doc_id == citation.doc_id and chunk.chunk_index == citation.chunk_index:
                return chunk
        return None
