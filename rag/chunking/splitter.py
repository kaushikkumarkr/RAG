from typing import List
from rag.ingestion.models import Document, Chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter

class BaseSplitter:
    def split(self, document: Document) -> List[Chunk]:
        raise NotImplementedError

class FixedSizeSplitter(BaseSplitter):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> List[Chunk]:
        text = document.content
        chunks = []
        if not text:
            return []
            
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_content = text[start:end]
            
            chunks.append(Chunk(
                doc_id=document.id,
                content=chunk_content,
                chunk_index=chunk_index,
                metadata=document.metadata
            ))
            
            chunk_index += 1
            start += (self.chunk_size - self.chunk_overlap)
            
        return chunks

class RecursiveSplitter(BaseSplitter):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
    
    def split(self, document: Document) -> List[Chunk]:
        texts = self.splitter.split_text(document.content)
        return [
            Chunk(
                doc_id=document.id,
                content=text,
                chunk_index=i,
                metadata=document.metadata
            )
            for i, text in enumerate(texts)
        ]
