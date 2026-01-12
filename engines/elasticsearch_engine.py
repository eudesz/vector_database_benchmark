from elasticsearch import Elasticsearch, helpers
from .base import BaseEngine
from typing import List, Dict, Any

class ElasticsearchEngine(BaseEngine):
    def __init__(self, host="localhost", port=9201, collection_name="benchmark"):
        # Note port 9201 for ES to avoid conflict with OS
        super().__init__(host, port, collection_name)
        self.client = None

    def init_client(self):
        self.client = Elasticsearch(
            f"http://{self.host}:{self.port}",
            request_timeout=30
        )

    def create_collection(self, dimension: int, config: Dict[str, Any] = None):
        config = config or {}
        m = config.get('m', 16)
        ef_construct = config.get('ef_construction', 100)

        index_body = {
            "settings": {
                "index": {
                    "refresh_interval": "-1" # Optimize for bulk insert
                }
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dimension,
                        "index": True,
                        "similarity": "cosine",
                        "index_options": {
                            "type": "hnsw",
                            "m": m,
                            "ef_construction": ef_construct
                        }
                    }
                }
            }
        }

        if self.client.indices.exists(index=self.collection_name):
            self.client.indices.delete(index=self.collection_name)
        
        self.client.indices.create(index=self.collection_name, body=index_body)

    def insert(self, vectors: List[List[float]], ids: List[int] = None):
        if ids is None:
            ids = list(range(len(vectors)))
            
        actions = []
        for i, vec in zip(ids, vectors):
            action = {
                "_index": self.collection_name,
                "_id": str(i),
                "_source": {
                    "embedding": vec
                }
            }
            actions.append(action)
            
            if len(actions) >= 1000:
                helpers.bulk(self.client, actions)
                actions = []
                
        if actions:
            helpers.bulk(self.client, actions)

    def refresh(self):
        self.client.indices.refresh(index=self.collection_name)
        self.client.indices.put_settings(
            index=self.collection_name, 
            body={"index": {"refresh_interval": "1s"}}
        )

    def search(self, query: List[float], k: int, search_params: Dict[str, Any] = None) -> List[int]:
        search_params = search_params or {}
        ef_search = search_params.get('ef_search', 100)
        # ES uses 'num_candidates' which must be >= k
        num_candidates = max(k, ef_search) 
        
        knn_query = {
            "field": "embedding",
            "query_vector": query,
            "k": k,
            "num_candidates": num_candidates
        }
        
        response = self.client.search(
            index=self.collection_name, 
            knn=knn_query,
            source=False
        )
        return [int(hit['_id']) for hit in response['hits']['hits']]

    def clean(self):
        if self.client and self.client.indices.exists(index=self.collection_name):
            self.client.indices.delete(index=self.collection_name)
