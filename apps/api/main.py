from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from apps.api.settings import settings
from apps.api.routers import ingest, search, ask
from apps.api.telemetry import setup_telemetry
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Starting RAG Foundry API...")
    setup_telemetry()
    yield
    # Shutdown logic
    print("Shutting down RAG Foundry API...")

app = FastAPI(
    title="RAG Foundry API",
    description="Local-first Advanced RAG Platform",
    version="0.1.0",
    lifespan=lifespan
)

FastAPIInstrumentor.instrument_app(app)

app.include_router(ingest.router)
app.include_router(search.router)
app.include_router(ask.router)

@app.get("/health")
async def health_check():
    """Health check endpoint to verify service status."""
    return {
        "status": "ok",
        "version": "0.1.0",
        "environment": settings.ENV
    }

@app.get("/")
async def root():
    return {"message": "Welcome to RAG Foundry"}

class AgentRequest(BaseModel):
    question: str

@app.post("/agent/ask")
async def ask_agent(request: AgentRequest):
    from rag.agent.runner import AgentRunner
    runner = AgentRunner()
    answer = runner.run(request.question)
    return {"answer": answer}

