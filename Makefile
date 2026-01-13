.PHONY: up down test lint setup clean ingest_sample build_index ask_demo eval health

VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
UVICORN = $(VENV_DIR)/bin/uvicorn

# Setup virtual environment and install dependencies
setup:
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Start Infrastructure Services (Qdrant, Phoenix)
services:
	docker compose up -d qdrant phoenix

# Start all services (including API in docker - optional)
up:
	docker compose up -d

# Stop Docker services
down:
	docker compose down

# Run all tests
test:
	PYTHONPATH=. $(PYTHON) -m pytest tests/ -v

# Run unit tests only
test-unit:
	PYTHONPATH=. $(PYTHON) -m pytest tests/unit -v

# Run with coverage
test-cov:
	PYTHONPATH=. $(PYTHON) -m pytest tests/ -v --cov=rag --cov-report=term-missing

# Run guardrails test script
test-guardrails:
	PYTHONPATH=. $(PYTHON) scripts/test_guardrails.py

# Run citations test script
test-citations:
	PYTHONPATH=. $(PYTHON) scripts/test_citations.py

# Run confidence test script  
test-confidence:
	PYTHONPATH=. $(PYTHON) scripts/test_confidence.py

# Run linting
lint:
	$(PYTHON) -m ruff check .

# Run API locally (needs Qdrant/Phoenix running via docker)
run-local:
	$(UVICORN) apps.api.main:app --reload --port 8000

# Health check
health:
	curl http://localhost:8000/health

# Clean artifacts
clean:
	rm -rf $(VENV_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

# Placeholders for future sprints
ingest_sample:
	curl -X POST "http://localhost:8000/ingest/local" \
		-H "Content-Type: application/json" \
		-d '{"path": "sample_data"}'

build_index:
	@echo "Not implemented yet (Sprint 1)"

ask_demo:
search_hybrid:
	curl -X POST "http://localhost:8000/search/hybrid" \
		-H "Content-Type: application/json" \
		-d '{"query": "Lisp", "top_k": 3, "alpha": 0.5}'

ask:
	curl -X POST "http://localhost:8000/ask" \
		-H "Content-Type: application/json" \
		-d '{"question": "How did Lisp start?", "use_hybrid": true}'

eval:
	PYTHONPATH=. .venv/bin/python scripts/run_eval.py


