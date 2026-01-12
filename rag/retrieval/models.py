from pydantic import BaseModel
from typing import List, Dict, Any

class ScoredChunk(BaseModel):
    content: str
    score: float
    doc_id: str
    chunk_index: int
    metadata: Dict[str, Any]
