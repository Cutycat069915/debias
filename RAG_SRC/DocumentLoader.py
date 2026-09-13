import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from glob import glob

from RAG_SRC.Document import Document, Chunk




class DocumentLoader:
    def __init__(self, supported_extensions: List[str] = [".txt", ".md", ".pdf"]):
        self.supported_extensions = supported_extensions


    def load_file(self, file_path: str) -> Document:
        """Reads a single file and extracts text and metadata."""
    
        path  = Path(file_path)
        if path.suffix not in self.supported_extensions:
            raise ValueError(f"Unsupported file extension: {path.suffix}")
        content = ""
        if path.suffix in (".txt" ,  ".md"):
            
            with open(path,  "r",  encoding="utf-8") as file:
                content =  file.read()
                
        elif path.suffix == ".pdf" :
            content  = ""
        metadata = {
            "source": str(path.resolve()),
            "filename": path.name,
            "extension": path.suffix
        }
        return Document(content=content, metadata=metadata)


    def load_directory(self, directory_path: str) -> List[Document]:
        documents = []
        dir_path = Path(directory_path)

        if not  dir_path.is_dir():
            raise NotADirectoryError(f"The path '{directory_path}' is not a valid directory.")
        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                try:
                    doc = self.load_file(file_path)
                    documents.append(doc)
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")
        return documents
                    
