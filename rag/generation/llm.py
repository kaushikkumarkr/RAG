from openai import OpenAI
from apps.api.settings import settings
import logging
from openinference.instrumentation.openai import OpenAIInstrumentor

logger = logging.getLogger(__name__)

# Instrument OpenAI global
OpenAIInstrumentor().instrument()

class LLMService:
    def __init__(self):
        # Point to local LM Studio or Ollama
        self.client = OpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key="lm-studio"  # Usually ignored by local runners
        )

    def generate_completion(self, system_prompt: str, user_message: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="mlx-community/Qwen2.5-7B-Instruct-4bit", 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM Generation Error: {e}")
            return f"Error generating answer: {e}"

    def chat(self, messages: list) -> str:
        try:
            response = self.client.chat.completions.create(
                model="mlx-community/Qwen2.5-7B-Instruct-4bit",
                messages=messages,
                temperature=0.5,
                stop=["Observation:"] # Stop generating before hallucinating an observation
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM Chat Error: {e}")
            return f"Error: {e}"
