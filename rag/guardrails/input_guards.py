"""
Input Guardrails: Validate user queries before processing.
- Off-topic detection
- Jailbreak/prompt injection detection
- PII detection
"""
import re
from typing import List, Optional
from rag.guardrails.models import GuardResult, GuardType, GuardAction

# Common jailbreak patterns
JAILBREAK_PATTERNS = [
    r"ignore (all |previous |your )?instructions",
    r"disregard (all |previous |your )?instructions",
    r"forget (all |previous |your )?instructions",
    r"you are now",
    r"pretend (to be|you are)",
    r"act as",
    r"roleplay as",
    r"bypass (the |your )?safety",
    r"override (the |your )?system",
    r"(system|developer) mode",
    r"DAN mode",
    r"jailbreak",
]

# PII detection patterns
PII_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}",
    "ssn": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
}

# Topics we expect (AI/ML research papers)
EXPECTED_TOPICS = [
    "transformer", "attention", "neural network", "deep learning",
    "machine learning", "gpt", "llm", "language model", "ai",
    "encoder", "decoder", "embedding", "training", "inference",
    "model", "architecture", "paper", "research"
]


class InputGuards:
    """Collection of input guardrails."""
    
    def __init__(self, embedding_service=None):
        """
        Args:
            embedding_service: Optional EmbeddingService for semantic checks.
        """
        self.embedding_service = embedding_service
        self._jailbreak_patterns = [re.compile(p, re.IGNORECASE) for p in JAILBREAK_PATTERNS]
        self._pii_patterns = {k: re.compile(v) for k, v in PII_PATTERNS.items()}
    
    def check_off_topic(self, query: str, threshold: float = 0.1) -> GuardResult:
        """
        Check if query is relevant to the knowledge base topics.
        Uses keyword matching (fast) or embedding similarity (accurate).
        """
        query_lower = query.lower()
        
        # Simple keyword check
        topic_score = sum(1 for topic in EXPECTED_TOPICS if topic in query_lower)
        normalized_score = min(1.0, topic_score / 3)  # 3+ keywords = fully on-topic
        
        is_off_topic = normalized_score < threshold
        
        return GuardResult(
            guard_type=GuardType.OFF_TOPIC,
            triggered=is_off_topic,
            action=GuardAction.WARN if is_off_topic else GuardAction.ALLOW,
            confidence=1 - normalized_score,  # High confidence it's off-topic if low score
            message="Query appears off-topic for this knowledge base" if is_off_topic else None,
            details={"topic_score": topic_score, "normalized": normalized_score}
        )
    
    def check_jailbreak(self, query: str) -> GuardResult:
        """
        Detect prompt injection / jailbreak attempts.
        """
        matches = []
        for pattern in self._jailbreak_patterns:
            if pattern.search(query):
                matches.append(pattern.pattern)
        
        is_jailbreak = len(matches) > 0
        confidence = min(1.0, len(matches) * 0.5)  # More patterns = higher confidence
        
        return GuardResult(
            guard_type=GuardType.JAILBREAK,
            triggered=is_jailbreak,
            action=GuardAction.BLOCK if is_jailbreak else GuardAction.ALLOW,
            confidence=confidence,
            message="Potential prompt injection detected" if is_jailbreak else None,
            details={"matched_patterns": matches}
        )
    
    def check_pii(self, query: str) -> GuardResult:
        """
        Detect PII (Personally Identifiable Information) in query.
        """
        found_pii = {}
        for pii_type, pattern in self._pii_patterns.items():
            matches = pattern.findall(query)
            if matches:
                found_pii[pii_type] = len(matches)
        
        has_pii = len(found_pii) > 0
        
        return GuardResult(
            guard_type=GuardType.PII,
            triggered=has_pii,
            action=GuardAction.WARN if has_pii else GuardAction.ALLOW,
            confidence=1.0 if has_pii else 0.0,
            message=f"PII detected in query: {list(found_pii.keys())}" if has_pii else None,
            details={"pii_found": found_pii}
        )
    
    def run_all(self, query: str) -> List[GuardResult]:
        """Run all input guardrails."""
        return [
            self.check_off_topic(query),
            self.check_jailbreak(query),
            self.check_pii(query),
        ]
