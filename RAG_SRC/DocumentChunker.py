from RAG_SRC.Document import Document, Chunk
from typing import List, Dict, Any
class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size to prevent infinite loops.")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    def split_document(self, document: Document) -> List[Chunk]:
        """Splits a single document into smaller chunks with overlap."""
        
        text =  document.content
        chunks  = []
        if not text:
           
            return []
        if len(text) <= self.chunk_size:
            return [Chunk(content = text, metadata =  document.metadata.copy(), ids = f"{document.metadata['filename']}_{0}")]
        start = 0 
        chunk_index = 0
        while start < len(text):
            end =  start + self.chunk_size

            chunk_text =  text[start:end]
            chunk_metadata = document.metadata.copy()
            chunk_metadata["chunk_index"] = chunk_index
            chunk_id = f"{document.metadata['filename']}_{chunk_index}"
            chunks.append(Chunk(content=chunk_text, metadata=chunk_metadata,ids=chunk_id))

            start += (self.chunk_size - self.chunk_overlap)
            chunk_index += 1
        return chunks

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        """Processes a batch of documents."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.split_document(doc))
        return all_chunks

