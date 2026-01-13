"""
Data models for guardrails.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

class GuardType(Enum):
    """Types of guardrails."""
    # Input guards
    OFF_TOPIC = "off_topic"
    JAILBREAK = "jailbreak"
    PII = "pii"
    # Output guards
    HALLUCINATION = "hallucination"
    TOXICITY = "toxicity"
    REFUSAL = "refusal"

class GuardAction(Enum):
    """Actions to take when a guard is triggered."""
    ALLOW = "allow"      # Let it through
    WARN = "warn"        # Allow but flag
    BLOCK = "block"      # Reject the request/response

@dataclass
class GuardResult:
    """Result from a single guardrail check."""
    guard_type: GuardType
    triggered: bool
    action: GuardAction
    confidence: float  # 0.0 to 1.0
    message: Optional[str] = None
    details: Optional[dict] = None
    
    def __repr__(self):
        status = "🚫 BLOCKED" if self.action == GuardAction.BLOCK else (
            "⚠️ WARNING" if self.action == GuardAction.WARN else "✅ ALLOWED"
        )
        return f"{status} [{self.guard_type.value}] conf={self.confidence:.2f}"

@dataclass
class GuardrailsResult:
    """Aggregated result from all guardrails."""
    passed: bool
    results: List[GuardResult]
    blocked_by: Optional[GuardType] = None
    
    @property
    def warnings(self) -> List[GuardResult]:
        return [r for r in self.results if r.action == GuardAction.WARN and r.triggered]
    
    @property
    def blocks(self) -> List[GuardResult]:
        return [r for r in self.results if r.action == GuardAction.BLOCK and r.triggered]
    
    def summary(self) -> str:
        if self.passed:
            if self.warnings:
                return f"⚠️ Passed with {len(self.warnings)} warning(s)"
            return "✅ All guardrails passed"
        return f"🚫 Blocked by {self.blocked_by.value if self.blocked_by else 'unknown'}"
