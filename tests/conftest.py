"""
Pytest configuration for RAG Foundry tests.
"""
import sys
import os
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Environment variables for testing
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("LLM_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:8080/v1")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "test-key")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "test-secret")
os.environ.setdefault("LANGFUSE_HOST", "http://localhost:3000")


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "What is the architecture of the Transformer model?"


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    from rag.retrieval.models import ScoredChunk
    return [
        ScoredChunk(
            content="The Transformer uses self-attention mechanisms.",
            score=0.9,
            doc_id="test-doc-1",
            chunk_index=0,
            metadata={"source": "test.pdf"}
        ),
        ScoredChunk(
            content="Attention allows the model to focus on relevant parts.",
            score=0.85,
            doc_id="test-doc-1",
            chunk_index=1,
            metadata={"source": "test.pdf"}
        ),
    ]


@pytest.fixture
def mock_answer():
    """Sample LLM answer."""
    return "The Transformer uses attention [test-doc-1:0] to process sequences."
