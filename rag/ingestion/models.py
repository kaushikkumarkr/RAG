from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

class Document(BaseModel):
    """
    Represents a raw document ingested into the system.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Chunk(BaseModel):
    """
    Represents a chunk of a document, ready for embedding and indexing.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    content: str
    vector: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_index: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
