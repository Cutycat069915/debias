from RAG_SRC.IngestionPipline import IngestionPipeline
from RAG_SRC.DocumentLoader import DocumentLoader
from RAG_SRC.DocumentChunker import DocumentChunker
from RAG_SRC.EmbeddingManager import EmbeddingManager_IP
from RAG_SRC.VectorStoreManager import VectorStoreManager
if __name__ == "__main__":
    pipeline = IngestionPipeline(
        loader=DocumentLoader(),
        chunker=DocumentChunker(chunk_size=800, chunk_overlap=150),
        embedder=EmbeddingManager_IP(),
        vector_store=VectorStoreManager(db_path="./chroma_db")
    )
    pipeline.run("./RAG")
    