from typing import List, Optional
from langfuse import Langfuse
from rag.retrieval.models import ScoredChunk
from rag.generation.llm import LLMService

# Initialize Langfuse for manual tracing
langfuse = Langfuse()

class GenerationService:
    def __init__(self):
        self.llm_service = LLMService()

    def _build_system_prompt(self) -> str:
        return (
            "You are a helpful AI assistant called RAG Foundry. "
            "You answer questions based STRICTLY on the provided context. "
            "If the answer is not in the context, say 'I cannot answer this based on the provided information.' "
            "Cite your sources using [doc_id:chunk_index] format at the end of sentences where appropriate."
        )

    def _build_prompts(self, query: str, chunks: List[ScoredChunk]) -> str:
        context_str = ""
        for i, chunk in enumerate(chunks):
            # Format: [doc_id:index] Content
            citation = f"[{chunk.doc_id}:{chunk.chunk_index}]"
            context_str += f"{citation} {chunk.content}\n\n"
            
        user_message = (
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        return user_message

    def generate_answer(self, query: str, chunks: List[ScoredChunk], observation=None) -> str:
        """
        Generate an answer using the LLM with provided context.
        
        Args:
            query: The user's question.
            chunks: Retrieved and reranked context chunks.
            observation: Optional Langfuse observation (trace/span) to nest under.
                        If provided, creates a child span. Otherwise, creates standalone trace.
        """
        # Create span (nested or standalone)
        if observation:
            span = observation.span(
                name="generation",
                input={"query": query, "num_chunks": len(chunks)}
            )
        else:
            span = langfuse.trace(
                name="generation",
                input={"query": query, "num_chunks": len(chunks)}
            )
        
        try:
            system_prompt = self._build_system_prompt()
            user_message = self._build_prompts(query, chunks)
            
            answer = self.llm_service.generate_completion(system_prompt, user_message)
            
            span.end(output={"answer": answer[:200] + "..." if len(answer) > 200 else answer})
            return answer
        except Exception as e:
            span.end(output={"error": str(e)})
            raise
