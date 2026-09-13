from typing import List, Dict, Any
from dataclasses import dataclass
from RAG_SRC.DocumentLoader import DocumentLoader
from RAG_SRC.DocumentChunker import DocumentChunker
from RAG_SRC.EmbeddingManager import EmbeddingManager
from RAG_SRC.VectorStoreManager import VectorStoreManager
class IngestionPipeline:
    def __init__(self, 
                 loader: DocumentLoader, 
                 chunker: DocumentChunker, 
                 embedder: EmbeddingManager, 
                 vector_store: VectorStoreManager):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def run(self, source_directory: str):
        """Executes the full ingestion workflow."""
        print(f"Starting ingestion from {source_directory}...")
        
       
        documents = self.loader.load_directory(source_directory)
        print(f"Loaded {len(documents)} documents.")
        
        chunks = self.chunker.split_documents(documents)
        print(f"Created {len(chunks)} chunks.")
        
        
        embedded_chunks = self.embedder.embed_chunks(chunks)
        print("Generated embeddings.")
        
        self.vector_store.add_chunks(embedded_chunks)
        print("Successfully stored in vector database.")
