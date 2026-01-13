# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG Foundry is a local-first Agentic RAG (Retrieval Augmented Generation) platform optimized for Apple Silicon via MLX. It implements a ReAct (Reason+Act) loop with hybrid retrieval combining dense vector search (Qdrant) and sparse keyword search (BM25).

## Common Commands

```bash
# Setup
make setup                    # Create venv and install dependencies

# Run services
make services                 # Start Qdrant + Phoenix in Docker
make up                       # Start all Docker services (API, UI, Qdrant, Phoenix)
make down                     # Stop Docker services
bash scripts/start_mlx.sh     # Start MLX inference server on port 8080 (run on host, not Docker)

# Development
make run-local                # Run API locally (requires Qdrant/Phoenix via make services)
make health                   # Check API health

# Testing & Quality
make test                     # Run pytest
make lint                     # Run ruff linter
make eval                     # Run RAGAS evaluation pipeline

# Cleanup
make clean                    # Remove venv and cache directories
```

## Architecture

### Core Modules (`rag/`)

- **`agent/`**: ReAct agent implementation
  - `runner.py`: Main agent loop with Thought/Action/Observation cycle
  - `tools.py`: SearchTool wrapper for retrieval
  - `decomposer.py`: Query decomposition

- **`retrieval/`**: Hybrid search orchestration
  - `service.py`: Combines dense (Qdrant) and sparse (BM25) search with score normalization
  - Alpha parameter controls dense vs sparse weight (0.0-1.0)

- **`ingestion/`**: Document processing pipeline
  - `service.py`: Load → Chunk → Embed → Index (dense + sparse)
  - `loaders.py`: LoaderFactory for PDF/MD/TXT files
  - Supports: `.txt`, `.md`, `.pdf`

- **`embeddings/`**: Vector embedding via sentence-transformers
- **`vector_store/`**: Qdrant client wrapper
- **`sparse/`**: BM25 index implementation
- **`rerank/`**: Cross-encoder reranking
- **`generation/`**: LLM service wrapper (OpenAI-compatible API)

### API Layer (`apps/api/`)

- FastAPI application with OpenTelemetry instrumentation
- Routers: `/ingest`, `/search`, `/ask`, `/agent/ask`
- Settings via pydantic-settings (`.env` or environment variables)

### Infrastructure (Docker Compose)

| Service | Port | Purpose |
|---------|------|---------|
| API | 8000 | FastAPI backend |
| UI | 8501 | Streamlit chat interface |
| Qdrant | 6333/6334 | Vector database |
| Phoenix | 6006 | Observability UI (traces) |
| MLX Server | 8080 | Local LLM (runs on host, not Docker) |

## Key Configuration

Environment variables (in `.env` or docker-compose):
- `QDRANT_URL`: Vector DB endpoint (default: `http://localhost:6333`)
- `LLM_BASE_URL`: MLX server endpoint (default: `http://localhost:8080/v1`)
- `PHOENIX_COLLECTOR_ENDPOINT`: OTel collector (default: `http://localhost:4317`)

## Evaluation

RAGAS metrics are configured in `eval/thresholds.yaml`:
- `faithfulness`: 0.7
- `answer_relevancy`: 0.7
- `context_precision`: 0.5
- `context_recall`: 0.5

Test questions in `eval/eval_questions.jsonl` and `eval/eval_questions_sanity.jsonl`.

## Development Notes

- MLX server must run natively on macOS host (not in Docker) for Apple Silicon optimization
- The agent uses a max of 5 reasoning steps before returning
- Hybrid search fetches 2x top_k candidates before merging and normalization
- BM25 index is rebuilt on each ingestion (append + rebuild pattern)
