from typing import List, Dict, Any
from dataclasses import dataclass
from glob import glob
import chromadb
from sentence_transformers import SentenceTransformer
# --- Data Structures ---
from RAG_SRC.Document import Document, Chunk
class VectorStoreManager:
    def __init__(self, db_path: str = "./local_vectordb", collection_name: str = "rag_collection",model_name: str = "./models/all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.model = SentenceTransformer(model_name)
    def add_chunks(self, chunks: List[Chunk]) -> bool:
        """Inserts text, metadata, and vectors into the database."""
        documents = []
        metadatas =  []
        ids = []
        embeddings =[]
        for chunk in chunks:
            documents.append(chunk.content)
            ids.append(chunk.ids)
            metadatas.append(chunk.metadata if hasattr(chunk, 'metadata') and chunk.metadata else {})
            embeddings.append(chunk.vector)
        try:
           
            if embeddings:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
            else:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            return True
            
        except Exception as e:
            print(f"Error inserting into vector store: {e}")
            return False
    def search(self, query: str, top_k: int = 5) -> List[Chunk]:
        try:
            query_vector =  self.model.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k
            )

            retrieved_chunks = []
            
            # Chroma returns lists of lists since it supports batch querying.
            # We take the 0th index because we only passed a single query_vector.
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    
                    # Reconstruct the Chunk object from the database response
                    chunk = Chunk(
                        content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i] if results['metadatas'] else {},
                        ids=results['ids'][0][i] 
                    )
                        
                    retrieved_chunks.append(chunk)

            return retrieved_chunks
            
        except Exception as e:
            print(f"Error during vector search: {e}")
            return []
