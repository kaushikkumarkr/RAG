from typing import List
from opentelemetry import trace
from rag.retrieval.models import ScoredChunk
from rag.generation.llm import LLMService

tracer = trace.get_tracer(__name__)

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

    @tracer.start_as_current_span("generate_answer")
    def generate_answer(self, query: str, chunks: List[ScoredChunk]) -> str:
        system_prompt = self._build_system_prompt()
        user_message = self._build_prompts(query, chunks)
        
        return self.llm_service.generate_completion(system_prompt, user_message)
