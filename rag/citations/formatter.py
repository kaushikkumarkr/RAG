"""
Citation Formatter: Convert citations to numbered references.
"""
import re
from typing import List, Dict
from rag.citations.models import Citation, SourceReference
from rag.retrieval.models import ScoredChunk


class CitationFormatter:
    """
    Format citations as numbered references [1], [2], etc.
    
    Also generates a "Sources" section for display.
    """
    
    def to_numbered(
        self, 
        text: str, 
        citations: List[Citation],
        chunks: List[ScoredChunk]
    ) -> tuple[str, List[SourceReference]]:
        """
        Replace [doc_id:chunk_index] citations with [1], [2], etc.
        
        Args:
            text: Original text with raw citations.
            citations: Extracted citations.
            chunks: Retrieved chunks for source info.
        
        Returns:
            Tuple of (formatted_text, source_references)
        """
        if not citations:
            return text, []
        
        # Create mapping from raw citation to reference number
        # Deduplicate by key (same source = same number)
        key_to_ref: Dict[str, int] = {}
        sources: List[SourceReference] = []
        ref_number = 1
        
        for citation in citations:
            if citation.key not in key_to_ref:
                key_to_ref[citation.key] = ref_number
                
                # Find the chunk for this citation
                chunk = self._find_chunk(citation, chunks)
                source_name = None
                snippet = None
                
                if chunk:
                    # Try to get source filename from metadata
                    source_name = chunk.metadata.get("source") or chunk.metadata.get("filename")
                    snippet = chunk.content[:150]
                
                sources.append(SourceReference(
                    ref_number=ref_number,
                    doc_id=citation.doc_id,
                    chunk_index=citation.chunk_index,
                    source_name=source_name,
                    snippet=snippet
                ))
                
                ref_number += 1
        
        # Replace all citations with numbered references
        formatted_text = text
        for citation in sorted(citations, key=lambda c: c.position, reverse=True):
            # Replace from end to preserve positions
            ref_num = key_to_ref[citation.key]
            formatted_text = (
                formatted_text[:citation.position] + 
                f"[{ref_num}]" + 
                formatted_text[citation.position + len(citation.raw_text):]
            )
        
        return formatted_text, sources
    
    def _find_chunk(self, citation: Citation, chunks: List[ScoredChunk]) -> ScoredChunk | None:
        """Find chunk matching citation."""
        for chunk in chunks:
            if chunk.doc_id == citation.doc_id and chunk.chunk_index == citation.chunk_index:
                return chunk
        return None
    
    def generate_sources_section(self, sources: List[SourceReference]) -> str:
        """
        Generate a formatted sources section.
        
        Returns:
            Sources:
            [1] filename.pdf (chunk 0)
                "Text preview..."
            [2] another.pdf (chunk 3)
                "Another preview..."
        """
        if not sources:
            return ""
        
        lines = ["", "---", "**Sources:**"]
        for source in sources:
            lines.append(source.format_full())
        
        return "\n".join(lines)
