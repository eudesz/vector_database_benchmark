from opensearchpy import OpenSearch, helpers
from .base import BaseEngine
from typing import List, Dict, Any

class OpenSearchEngine(BaseEngine):
    def __init__(self, host="localhost", port=9200, collection_name="benchmark"):
        super().__init__(host, port, collection_name)
        self.client = None

    def init_client(self):
        self.client = OpenSearch(
            hosts=[{'host': self.host, 'port': self.port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False
        )

    def create_collection(self, dimension: int, config: Dict[str, Any] = None):
        config = config or {}
        m = config.get('m', 16)
        ef_construct = config.get('ef_construction', 100)

        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100, # Default
                    "refresh_interval": "-1" # Optimize for bulk insert
                }
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": dimension,
                        "method": {
                            "name": "hnsw",
                            "engine": "lucene",
                            "space_type": "cosinesimil",
                            "parameters": {
                                "m": m,
                                "ef_construction": ef_construct
                            }
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
        # Restore refresh interval for search
        self.client.indices.put_settings(
            index=self.collection_name, 
            body={"index": {"refresh_interval": "1s"}}
        )

    def search(self, query: List[float], k: int, search_params: Dict[str, Any] = None) -> List[int]:
        search_params = search_params or {}
        ef_search = search_params.get('ef_search', 0)
        
        # OpenSearch ef_search is index setting, but can be updated dynamic?
        # Ideally we don't update settings per query, but for benchmark maybe.
        # Alternatively, newer OpenSearch versions might support it in query? No, usually index setting.
        # We will skip per-query setting here for speed, or update if critical.
        
        body = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query,
                        "k": k
                    }
                }
            },
            "_source": False 
        }
        
        response = self.client.search(index=self.collection_name, body=body)
        return [int(hit['_id']) for hit in response['hits']['hits']]

    def clean(self):
        if self.client and self.client.indices.exists(index=self.collection_name):
            self.client.indices.delete(index=self.collection_name)
