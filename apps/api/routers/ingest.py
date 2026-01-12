from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
import shutil
import os
import tempfile
from rag.ingestion.service import IngestionService

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

class IngestResponse(BaseModel):
    message: str
    chunks_indexed: int

@router.post("/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    service = IngestionService()
    
    # Save uploaded file typically to a temp location
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
        
    try:
        chunks_count = service.ingest_file(tmp_path)
        return IngestResponse(message=f"Successfully ingested {file.filename}", chunks_indexed=chunks_count)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

class LocalIngestRequest(BaseModel):
    path: str

@router.post("/local", response_model=IngestResponse)
async def ingest_local(request: LocalIngestRequest):
    service = IngestionService()
    
    if not os.path.exists(request.path):
        raise HTTPException(status_code=404, detail="Path not found")
        
    if os.path.isfile(request.path):
        count = service.ingest_file(request.path)
    else:
        count = service.ingest_directory(request.path)
        
    return IngestResponse(message=f"Ingested from {request.path}", chunks_indexed=count)
