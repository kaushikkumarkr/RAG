import os
from typing import List, Dict, Any
from pathlib import Path
from rag.ingestion.models import Document
import pypdf

class BaseLoader:
    def load(self, file_path: str) -> List[Document]:
        raise NotImplementedError

class TextLoader(BaseLoader):
    def load(self, file_path: str) -> List[Document]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return [Document(
            content=content,
            source=file_path,
            metadata={"type": "text"}
        )]

class MarkdownLoader(BaseLoader):
    def load(self, file_path: str) -> List[Document]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        return [Document(
            content=content,
            source=file_path,
            metadata={"type": "markdown"}
        )]

class PDFLoader(BaseLoader):
    def load(self, file_path: str) -> List[Document]:
        reader = pypdf.PdfReader(file_path)
        content = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                content += text + "\n\n"
        
        return [Document(
            content=content,
            source=file_path,
            metadata={"type": "pdf", "pages": len(reader.pages)}
        )]

class LoaderFactory:
    @staticmethod
    def get_loader(file_path: str) -> BaseLoader:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".txt":
            return TextLoader()
        elif ext == ".md":
            return MarkdownLoader()
        elif ext == ".pdf":
            return PDFLoader()
        else:
            raise ValueError(f"Unsupported file type: {ext}")
