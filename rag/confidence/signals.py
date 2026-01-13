"""
Confidence Signals: Individual confidence calculations.
"""
import re
from typing import List
from rag.retrieval.models import ScoredChunk
from rag.citations.models import CitationResult

# Refusal patterns
REFUSAL_PATTERNS = [
    r"i cannot answer",
    r"i'm unable to",
    r"i am unable to",
    r"i don't have (enough )?information",
    r"cannot provide",
    r"not (mentioned|specified|provided) in",
]


class ConfidenceSignals:
    """
    Calculate individual confidence signals for RAG answers.
    
    Signals:
    1. Retrieval Quality: Based on rerank scores
    2. Source Agreement: Do chunks agree with each other?
    3. Citation Density: Number of citations per response length
    4. Refusal Check: Did the LLM refuse to answer?
    """
    
    def __init__(self):
        self._refusal_patterns = [re.compile(p, re.IGNORECASE) for p in REFUSAL_PATTERNS]
    
    def retrieval_confidence(self, chunks: List[ScoredChunk]) -> float:
        """
        Calculate confidence based on retrieval/rerank scores.
        
        Higher rerank scores = more confident the chunks are relevant.
        """
        if not chunks:
            return 0.0
        
        scores = [c.score for c in chunks]
        avg_score = sum(scores) / len(scores)
        
        # Normalize: Rerank scores can be negative or >1
        # Typical range is -5 to +10, normalize to 0-1
        normalized = (avg_score + 5) / 15
        return max(0.0, min(1.0, normalized))
    
    def source_agreement(self, chunks: List[ScoredChunk]) -> float:
        """
        Calculate how much the sources "agree".
        
        Simple heuristic: If chunks have similar content vocabulary,
        they're more likely to support a consistent answer.
        """
        if len(chunks) < 2:
            return 1.0  # Single source = no disagreement
        
        # Extract word sets from each chunk
        word_sets = []
        for chunk in chunks:
            words = set(chunk.content.lower().split())
            # Filter to meaningful words (>3 chars)
            words = {w for w in words if len(w) > 3}
            word_sets.append(words)
        
        # Calculate pairwise Jaccard similarity
        similarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                if union > 0:
                    similarities.append(intersection / union)
        
        if not similarities:
            return 0.5
        
        return sum(similarities) / len(similarities)
    
    def citation_density(self, answer: str, citation_result: CitationResult) -> float:
        """
        Calculate citation density: more citations = more grounded answer.
        """
        if not answer:
            return 0.0
        
        word_count = len(answer.split())
        citation_count = citation_result.valid_count
        
        # Target: ~1 citation per 50 words is ideal
        ideal_ratio = word_count / 50
        if ideal_ratio == 0:
            return 1.0 if citation_count > 0 else 0.0
        
        ratio = citation_count / max(ideal_ratio, 1)
        return min(1.0, ratio)
    
    def refusal_check(self, answer: str) -> float:
        """
        Check if the LLM refused to answer.
        
        Returns:
            1.0 if no refusal detected
            0.0 if refusal detected
        """
        answer_lower = answer.lower()
        
        for pattern in self._refusal_patterns:
            if pattern.search(answer_lower):
                return 0.0
        
        return 1.0
    
    def answer_length_check(self, answer: str) -> float:
        """
        Very short answers may indicate low confidence.
        
        Returns:
            1.0 for answers > 100 chars
            0.5 for answers 50-100 chars
            0.3 for answers < 50 chars
        """
        length = len(answer)
        if length > 100:
            return 1.0
        elif length > 50:
            return 0.7
        else:
            return 0.4
