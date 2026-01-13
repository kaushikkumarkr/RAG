"""
Unified Guardrail Service: Combines input and output guardrails.
"""
from typing import List, Optional
from langfuse import Langfuse
from rag.guardrails.models import GuardResult, GuardrailsResult, GuardType, GuardAction
from rag.guardrails.input_guards import InputGuards
from rag.guardrails.output_guards import OutputGuards
from rag.retrieval.models import ScoredChunk

langfuse = Langfuse()


class GuardrailService:
    """
    Unified service for running all guardrails.
    
    Usage:
        service = GuardrailService()
        
        # Check input before processing
        input_result = service.check_input(query)
        if not input_result.passed:
            return "I cannot process this query."
        
        # ... run RAG pipeline ...
        
        # Check output before returning
        output_result = service.check_output(answer, chunks)
        if not output_result.passed:
            return "I cannot provide this response."
    """
    
    def __init__(self):
        self.input_guards = InputGuards()
        self.output_guards = OutputGuards()
    
    def check_input(self, query: str, observation=None) -> GuardrailsResult:
        """
        Run all input guardrails on a query.
        
        Args:
            query: User's input query.
            observation: Optional Langfuse observation for logging.
        
        Returns:
            GuardrailsResult with pass/fail status and details.
        """
        # Create span if observation provided
        span = None
        if observation:
            span = observation.span(
                name="input_guardrails",
                input={"query": query[:100]}
            )
        
        results = self.input_guards.run_all(query)
        
        # Check for blocks
        blocked_by = None
        for result in results:
            if result.triggered and result.action == GuardAction.BLOCK:
                blocked_by = result.guard_type
                break
        
        passed = blocked_by is None
        
        guardrails_result = GuardrailsResult(
            passed=passed,
            results=results,
            blocked_by=blocked_by
        )
        
        if span:
            span.end(output={
                "passed": passed,
                "blocked_by": blocked_by.value if blocked_by else None,
                "warnings": len(guardrails_result.warnings)
            })
        
        return guardrails_result
    
    def check_output(
        self, 
        answer: str, 
        chunks: Optional[List[ScoredChunk]] = None,
        observation=None
    ) -> GuardrailsResult:
        """
        Run all output guardrails on an answer.
        
        Args:
            answer: LLM-generated answer.
            chunks: Retrieved chunks for hallucination checking.
            observation: Optional Langfuse observation for logging.
        
        Returns:
            GuardrailsResult with pass/fail status and details.
        """
        span = None
        if observation:
            span = observation.span(
                name="output_guardrails",
                input={"answer_length": len(answer)}
            )
        
        results = self.output_guards.run_all(answer, chunks)
        
        # Check for blocks
        blocked_by = None
        for result in results:
            if result.triggered and result.action == GuardAction.BLOCK:
                blocked_by = result.guard_type
                break
        
        passed = blocked_by is None
        
        guardrails_result = GuardrailsResult(
            passed=passed,
            results=results,
            blocked_by=blocked_by
        )
        
        if span:
            span.end(output={
                "passed": passed,
                "blocked_by": blocked_by.value if blocked_by else None,
                "warnings": len(guardrails_result.warnings)
            })
        
        return guardrails_result
    
    def format_block_message(self, result: GuardrailsResult) -> str:
        """Generate a user-friendly message when blocked."""
        if result.passed:
            return ""
        
        block_messages = {
            GuardType.JAILBREAK: "I cannot process requests that attempt to override my instructions.",
            GuardType.TOXICITY: "I cannot provide a response that may contain harmful content.",
            GuardType.OFF_TOPIC: "This query appears to be outside my knowledge domain.",
            GuardType.PII: "I've detected sensitive personal information in your query.",
            GuardType.HALLUCINATION: "I cannot verify the sources in this response.",
        }
        
        if result.blocked_by:
            return block_messages.get(
                result.blocked_by, 
                "I cannot process this request due to safety constraints."
            )
        return "I cannot process this request."
