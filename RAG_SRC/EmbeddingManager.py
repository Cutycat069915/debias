from typing import List, Dict, Any
from dataclasses import dataclass
import requests ,  json
from sentence_transformers import SentenceTransformer
# --- Data Structures ---
from RAG_SRC.Document import Document, Chunk
from typing import List, Dict, Any
class EmbeddingManager:
    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def embed_text(self, text: str) -> List[float]:
        """Calls the local Ollama API to get vector embeddings for a string."""
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model" : self.model_name,
            "prompt" : text
        }
        try:
            response = requests.post(url,  json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("embedding",[])
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Is the Ollama app running?"
            )
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while fetching embeddings: {e}")
            return []
        


    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Populates the 'vector' field for a list of Chunk objects."""
        for i , chunk in enumerate(chunks):
            if not chunk.content.strip():
                continue
            chunk.vector = self.embed_text(chunk.content)
            if (i+1) %10 ==0 :
                print(f"Embedded {i + 1}/{len(chunks)} chunks...")
        pass
class EmbeddingManager_IP:
    def __init__(self, model_name: str = "./models/all-MiniLM-L6-v2"):
        print(f"Loading local model '{model_name}' into memory. This may take a moment...")

        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully.")

    def embed_text(self, text: str) -> List[float]:
        """Generates vector embeddings for a single string locally."""
        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Populates the 'vector' field for a list of Chunk objects using batch processing."""
        valid_chunks = [c for c in chunks if c.content.strip()]
        
        if not valid_chunks:
            return chunks

        texts = [chunk.content for chunk in valid_chunks]
        
        print(f"Generating embeddings for {len(texts)} chunks locally...")
        
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        for chunk, vector in zip(valid_chunks, embeddings):
            chunk.vector = vector.tolist()
            
        return chunks