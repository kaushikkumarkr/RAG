from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    ENV: str = "development"
    QDRANT_URL: str = "http://localhost:6333"
    PHOENIX_COLLECTOR_ENDPOINT: str = "http://localhost:4317"
    LLM_BASE_URL: str = "http://localhost:8080/v1"  # MLX Server
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "http://localhost:3000"
    
    class Config:
        env_file = ".env"

settings = Settings()
