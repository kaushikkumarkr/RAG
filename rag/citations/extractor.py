"""
Citation Extractor: Parse citations from LLM output.
"""
import re
from typing import List
from rag.citations.models import Citation

# Citation pattern: [doc_id:chunk_index] where doc_id is alphanumeric/uuid and chunk_index is number
CITATION_PATTERN = r'\[([a-zA-Z0-9_-]+):(\d+)\]'


class CitationExtractor:
    """
    Extract citations from LLM-generated text.
    
    Example:
        text = "The Transformer uses attention [abc123:0] mechanisms."
        extractor = CitationExtractor()
        citations = extractor.extract(text)
        # [Citation(doc_id='abc123', chunk_index=0, position=32)]
    """
    
    def __init__(self):
        self._pattern = re.compile(CITATION_PATTERN)
    
    def extract(self, text: str) -> List[Citation]:
        """
        Extract all citations from text.
        
        Args:
            text: LLM-generated answer text.
        
        Returns:
            List of Citation objects with positions.
        """
        citations = []
        
        for match in self._pattern.finditer(text):
            doc_id = match.group(1)
            chunk_index = int(match.group(2))
            position = match.start()
            raw_text = match.group(0)
            
            citations.append(Citation(
                doc_id=doc_id,
                chunk_index=chunk_index,
                position=position,
                raw_text=raw_text
            ))
        
        return citations
    
    def has_citations(self, text: str) -> bool:
        """Check if text contains any citations."""
        return bool(self._pattern.search(text))
    
    def count_citations(self, text: str) -> int:
        """Count citations in text."""
        return len(self._pattern.findall(text))
