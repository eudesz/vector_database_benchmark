from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseEngine(ABC):
    def __init__(self, host: str, port: int, collection_name: str):
        self.host = host
        self.port = port
        self.collection_name = collection_name

    @abstractmethod
    def init_client(self):
        """Initialize the database client."""
        pass

    @abstractmethod
    def create_collection(self, dimension: int, config: Dict[str, Any] = None):
        """Create a collection/index with specific configuration."""
        pass

    @abstractmethod
    def insert(self, vectors: List[List[float]], ids: List[int] = None):
        """Insert vectors into the collection."""
        pass

    @abstractmethod
    def search(self, query: List[float], k: int, search_params: Dict[str, Any] = None) -> List[int]:
        """Search for the k nearest neighbors. Returns list of IDs."""
        pass

    @abstractmethod
    def clean(self):
        """Delete collection and clean up."""
        pass
    
    @abstractmethod
    def refresh(self):
        """Force refresh/commit (useful for Lucene based engines)."""
        pass
