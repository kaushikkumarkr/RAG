"""
Data models for citations.
"""
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class Citation:
    """A single citation extracted from LLM output."""
    doc_id: str
    chunk_index: int
    position: int  # Character position in original text
    raw_text: str  # The original citation text, e.g., "[abc123:0]"
    
    @property
    def key(self) -> str:
        """Unique key for this citation."""
        return f"{self.doc_id}:{self.chunk_index}"
    
    def __repr__(self):
        return f"Citation({self.key})"

@dataclass
class SourceReference:
    """A formatted source reference for display."""
    ref_number: int  # [1], [2], etc.
    doc_id: str
    chunk_index: int
    source_name: Optional[str] = None  # Filename or title
    snippet: Optional[str] = None  # Text preview
    
    def format_short(self) -> str:
        """Short format: [1]"""
        return f"[{self.ref_number}]"
    
    def format_full(self) -> str:
        """Full format for sources section."""
        source = self.source_name or self.doc_id[:8]
        snippet_preview = (self.snippet[:100] + "...") if self.snippet and len(self.snippet) > 100 else self.snippet
        return f"[{self.ref_number}] {source} (chunk {self.chunk_index})\n    \"{snippet_preview}\""

@dataclass
class CitationResult:
    """Result of citation extraction and validation."""
    citations: List[Citation] = field(default_factory=list)
    valid_citations: List[Citation] = field(default_factory=list)
    phantom_citations: List[Citation] = field(default_factory=list)  # Hallucinated refs
    sources: List[SourceReference] = field(default_factory=list)
    formatted_answer: Optional[str] = None  # Answer with [1], [2], etc.
    
    @property
    def has_phantoms(self) -> bool:
        return len(self.phantom_citations) > 0
    
    @property
    def citation_count(self) -> int:
        return len(self.citations)
    
    @property
    def valid_count(self) -> int:
        return len(self.valid_citations)
    
    def summary(self) -> str:
        if self.has_phantoms:
            return f"⚠️ {len(self.phantom_citations)} phantom citation(s) detected"
        return f"✅ {self.valid_count} valid citation(s)"
