from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)
from .base import BaseEngine
from typing import List, Dict, Any

class MilvusEngine(BaseEngine):
    def __init__(self, host="127.0.0.1", port=19530, collection_name="benchmark"):
        super().__init__(host, port, collection_name)
        self.collection = None

    def init_client(self):
        try:
            connections.connect("default", host=self.host, port=self.port, timeout=30)
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            raise

    def create_collection(self, dimension: int, config: Dict[str, Any] = None):
        config = config or {}
        m = config.get('m', 16)
        ef_construct = config.get('ef_construction', 200)

        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
        ]
        schema = CollectionSchema(fields, "Benchmark collection")
        self.collection = Collection(self.collection_name, schema)

        # Create Index
        index_params = {
            "metric_type": "L2", # Using L2 for normalized vectors (equivalent to Cosine ranking)
            "index_type": "HNSW",
            "params": {"M": m, "efConstruction": ef_construct}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)

    def insert(self, vectors: List[List[float]], ids: List[int] = None):
        if ids is None:
            ids = list(range(len(vectors)))
        
        # Milvus inserts are columnar: [ [ids...], [vectors...] ]
        data = [
            ids,
            vectors
        ]
        self.collection.insert(data)

    def refresh(self):
        if self.collection:
            self.collection.flush()
            self.collection.load() # Load into memory for search

    def search(self, query: List[float], k: int, search_params: Dict[str, Any] = None) -> List[int]:
        search_params = search_params or {}
        ef = search_params.get('ef_search', 64)
        
        param = {
            "metric_type": "L2",
            "params": {"ef": ef}
        }
        
        # Milvus search expects list of queries
        results = self.collection.search(
            data=[query],
            anns_field="embedding",
            param=param,
            limit=k,
            expr=None
        )
        
        return results[0].ids

    def clean(self):
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
        except Exception as e:
            print(f"Error cleaning Milvus: {e}")
            pass
