from qdrant_client import QdrantClient
from qdrant_client.http import models
from .base import BaseEngine
import time
from typing import List, Dict, Any

class QdrantEngine(BaseEngine):
    def __init__(self, host="localhost", port=6333, collection_name="benchmark"):
        super().__init__(host, port, collection_name)
        self.client = None

    def init_client(self):
        self.client = QdrantClient(host=self.host, port=self.port)

    def create_collection(self, dimension: int, config: Dict[str, Any] = None):
        config = config or {}
        
        # Extract HNSW params
        m = config.get('m', 16)
        ef_construct = config.get('ef_construction', 100)
        
        # Check if collection exists and recreate
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE
            ),
            hnsw_config=models.HnswConfigDiff(
                m=m,
                ef_construct=ef_construct
            )
        )

    def insert(self, vectors: List[List[float]], ids: List[int] = None):
        if ids is None:
            ids = list(range(len(vectors)))
            
        # Batch insert
        batch_size = 1000
        total = len(vectors)
        
        for i in range(0, total, batch_size):
            batch_vectors = vectors[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=models.Batch(
                    ids=batch_ids,
                    vectors=batch_vectors
                )
            )

    def refresh(self):
        # Qdrant is real-time, but for consistent benchmarking we might want to wait for indexing
        # Generally not strictly needed like ES refresh, but we can verify count
        pass

    def search(self, query: List[float], k: int, search_params: Dict[str, Any] = None) -> List[int]:
        search_params = search_params or {}
        ef_search = search_params.get('ef_search', 0) # 0 means default
        
        params = None
        if ef_search > 0:
            params = models.SearchParams(hnsw_ef=ef_search)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query,
            limit=k,
            search_params=params
        ).points
        
        return [hit.id for hit in results]

    def clean(self):
        if self.client and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
