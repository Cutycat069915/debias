from typing import List, Dict, Any
from dataclasses import dataclass
from glob import glob
@dataclass
class Chunk:
    """Represents a single piece of text ready for the vector database."""
    content: str
    metadata: Dict[str, Any]
    ids : str
    vector: List[float] = None
    

@dataclass
class Document:
    """Represents a full loaded document."""
    content: str
    metadata: Dict[str, Any]