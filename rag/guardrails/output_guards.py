"""
Output Guardrails: Validate LLM responses before returning to user.
- Hallucination detection (phantom citations)
- Toxicity/harmful content filtering
- Refusal detection
"""
import re
from typing import List, Optional
from rag.guardrails.models import GuardResult, GuardType, GuardAction
from rag.retrieval.models import ScoredChunk

# Toxic/harmful words and phrases
TOXIC_PATTERNS = [
    # Violence
    r"\b(kill|murder|attack|destroy|harm|hurt)\b",
    # Hate speech indicators
    r"\b(hate|hatred|racist|sexist)\b",
    # Dangerous instructions
    r"\b(bomb|weapon|explosive|poison)\b",
    # Self-harm
    r"\b(suicide|self-harm|cut yourself)\b",
]

# Refusal patterns
REFUSAL_PATTERNS = [
    r"i cannot answer",
    r"i'm unable to",
    r"i am unable to",
    r"i don't have (enough )?information",
    r"i cannot provide",
    r"i'm not able to",
    r"the (provided |given )?context (does not|doesn't)",
    r"based on the (provided |given )?information",
    r"not (mentioned|specified|provided) in the (context|information)",
]

# Citation pattern: [doc_id:chunk_index] or [uuid:number]
CITATION_PATTERN = r'\[([a-f0-9-]+):(\d+)\]'


class OutputGuards:
    """Collection of output guardrails."""
    
    def __init__(self):
        self._toxic_patterns = [re.compile(p, re.IGNORECASE) for p in TOXIC_PATTERNS]
        self._refusal_patterns = [re.compile(p, re.IGNORECASE) for p in REFUSAL_PATTERNS]
        self._citation_pattern = re.compile(CITATION_PATTERN)
    
    def check_hallucination(
        self, 
        answer: str, 
        chunks: List[ScoredChunk]
    ) -> GuardResult:
        """
        Detect phantom citations (references to sources that don't exist).
        """
        # Extract all citations from answer
        citations = self._citation_pattern.findall(answer)
        
        if not citations:
            # No citations = can't verify, but not necessarily hallucination
            return GuardResult(
                guard_type=GuardType.HALLUCINATION,
                triggered=False,
                action=GuardAction.WARN,
                confidence=0.3,
                message="No citations found in answer",
                details={"citations_found": 0}
            )
        
        # Build set of valid citation keys from chunks
        valid_citations = set()
        for chunk in chunks:
            key = f"{chunk.doc_id}:{chunk.chunk_index}"
            valid_citations.add(key)
        
        # Check each citation
        phantom_citations = []
        valid_count = 0
        for doc_id, chunk_idx in citations:
            key = f"{doc_id}:{chunk_idx}"
            if key not in valid_citations:
                phantom_citations.append(key)
            else:
                valid_count += 1
        
        has_phantoms = len(phantom_citations) > 0
        total_citations = len(citations)
        phantom_ratio = len(phantom_citations) / total_citations if total_citations > 0 else 0
        
        return GuardResult(
            guard_type=GuardType.HALLUCINATION,
            triggered=has_phantoms,
            action=GuardAction.WARN if has_phantoms else GuardAction.ALLOW,
            confidence=phantom_ratio,
            message=f"Found {len(phantom_citations)} phantom citation(s)" if has_phantoms else None,
            details={
                "total_citations": total_citations,
                "valid_citations": valid_count,
                "phantom_citations": phantom_citations
            }
        )
    
    def check_toxicity(self, answer: str) -> GuardResult:
        """
        Detect toxic or harmful content in the answer.
        """
        matches = []
        for pattern in self._toxic_patterns:
            found = pattern.findall(answer)
            if found:
                matches.extend(found)
        
        is_toxic = len(matches) > 0
        confidence = min(1.0, len(matches) * 0.3)
        
        return GuardResult(
            guard_type=GuardType.TOXICITY,
            triggered=is_toxic,
            action=GuardAction.BLOCK if is_toxic else GuardAction.ALLOW,
            confidence=confidence,
            message="Potentially harmful content detected" if is_toxic else None,
            details={"matched_terms": list(set(matches))}
        )
    
    def check_refusal(self, answer: str) -> GuardResult:
        """
        Detect when the LLM refused to answer.
        """
        matches = []
        for pattern in self._refusal_patterns:
            if pattern.search(answer):
                matches.append(pattern.pattern)
        
        is_refusal = len(matches) > 0
        confidence = min(1.0, len(matches) * 0.4)
        
        return GuardResult(
            guard_type=GuardType.REFUSAL,
            triggered=is_refusal,
            action=GuardAction.WARN if is_refusal else GuardAction.ALLOW,
            confidence=confidence,
            message="LLM refused to answer (may lack relevant context)" if is_refusal else None,
            details={"matched_patterns": len(matches)}
        )
    
    def run_all(self, answer: str, chunks: Optional[List[ScoredChunk]] = None) -> List[GuardResult]:
        """Run all output guardrails."""
        results = [
            self.check_toxicity(answer),
            self.check_refusal(answer),
        ]
        
        if chunks:
            results.append(self.check_hallucination(answer, chunks))
        
        return results
