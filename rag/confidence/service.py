"""
Confidence Service: Aggregate multi-signal confidence score.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from langfuse import Langfuse
from rag.confidence.signals import ConfidenceSignals
from rag.retrieval.models import ScoredChunk
from rag.citations.models import CitationResult

langfuse = Langfuse()

# Signal weights for aggregation
SIGNAL_WEIGHTS = {
    "retrieval": 0.30,    # 30% - How good are the retrieved chunks?
    "agreement": 0.20,    # 20% - Do sources agree?
    "citations": 0.20,    # 20% - Is answer well-cited?
    "refusal": 0.15,      # 15% - Did LLM refuse?
    "length": 0.15,       # 15% - Answer length check
}


@dataclass
class ConfidenceResult:
    """Result of confidence calculation."""
    score: float  # Aggregated 0.0-1.0 score
    breakdown: Dict[str, float] = field(default_factory=dict)
    level: str = "medium"  # low, medium, high
    disclaimer: Optional[str] = None
    
    def __post_init__(self):
        # Set level based on score
        if self.score >= 0.7:
            self.level = "high"
        elif self.score >= 0.4:
            self.level = "medium"
        else:
            self.level = "low"
            self.disclaimer = "⚠️ Low confidence: This answer may not be reliable."
    
    def __repr__(self):
        return f"Confidence({self.score:.2f}, level={self.level})"


class ConfidenceService:
    """
    Calculate aggregated confidence score from multiple signals.
    
    Usage:
        service = ConfidenceService()
        result = service.calculate(answer, chunks, citation_result)
        
        print(f"Confidence: {result.score:.2f} ({result.level})")
        if result.disclaimer:
            print(result.disclaimer)
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.signals = ConfidenceSignals()
        self.weights = weights or SIGNAL_WEIGHTS
    
    def calculate(
        self,
        answer: str,
        chunks: List[ScoredChunk],
        citation_result: Optional[CitationResult] = None,
        observation=None
    ) -> ConfidenceResult:
        """
        Calculate aggregated confidence score.
        
        Args:
            answer: LLM-generated answer.
            chunks: Retrieved/reranked chunks.
            citation_result: Optional citation processing result.
            observation: Optional Langfuse observation for logging.
        
        Returns:
            ConfidenceResult with aggregated score and breakdown.
        """
        span = None
        if observation:
            span = observation.span(
                name="confidence_scoring",
                input={"answer_length": len(answer), "num_chunks": len(chunks)}
            )
        
        # Calculate individual signals
        breakdown = {}
        
        # 1. Retrieval Quality
        breakdown["retrieval"] = self.signals.retrieval_confidence(chunks)
        
        # 2. Source Agreement
        breakdown["agreement"] = self.signals.source_agreement(chunks)
        
        # 3. Citation Density (if available)
        if citation_result:
            breakdown["citations"] = self.signals.citation_density(answer, citation_result)
        else:
            breakdown["citations"] = 0.5  # Neutral if no citation info
        
        # 4. Refusal Check
        breakdown["refusal"] = self.signals.refusal_check(answer)
        
        # 5. Answer Length
        breakdown["length"] = self.signals.answer_length_check(answer)
        
        # Aggregate with weights
        total_score = 0.0
        for signal_name, signal_score in breakdown.items():
            weight = self.weights.get(signal_name, 0.1)
            total_score += signal_score * weight
        
        # Normalize to ensure 0-1 range
        total_score = max(0.0, min(1.0, total_score))
        
        result = ConfidenceResult(
            score=round(total_score, 2),
            breakdown=breakdown
        )
        
        if span:
            span.end(output={
                "confidence_score": result.score,
                "level": result.level,
                **breakdown
            })
        
        return result
    
    def should_add_disclaimer(self, result: ConfidenceResult) -> bool:
        """Check if a disclaimer should be added."""
        return result.score < 0.5
    
    def format_disclaimer(self, result: ConfidenceResult) -> str:
        """Generate appropriate disclaimer based on confidence level."""
        if result.score < 0.3:
            return "⚠️ **Very Low Confidence**: This answer may not be reliable. Please verify with other sources."
        elif result.score < 0.5:
            return "⚠️ **Low Confidence**: Take this answer with caution."
        return ""
