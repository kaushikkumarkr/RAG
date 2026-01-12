from typing import List
from rag.generation.llm import LLMService

DECOMPOSER_SYSTEM_PROMPT = """
You are an expert at breaking down complex questions into simple, independent sub-questions.
Your goal is to generate a list of sub-questions that, when answered, will allow you to answer the original question.
Return ONLY the sub-questions, one per line. Do not number them.
Example:
Q: Compare X and Y.
Sub-Q:
What is X?
What is Y?
"""

class QueryDecomposer:
    def __init__(self):
        self.llm = LLMService()

    def decompose(self, query: str) -> List[str]:
        response = self.llm.generate_completion(
            system_prompt=DECOMPOSER_SYSTEM_PROMPT,
            user_message=query
        )
        questions = [line.strip() for line in response.split('\n') if line.strip()]
        return questions
