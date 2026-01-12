import weaviate
import weaviate.classes.config as wvc
from .base import BaseEngine
from typing import List, Dict, Any

class WeaviateEngine(BaseEngine):
    def __init__(self, host="localhost", port=8080, collection_name="Benchmark"):
        # Weaviate usually runs on 8080
        # collection_name in Weaviate must be Capitalized Class Name
        super().__init__(host, port, collection_name.capitalize())
        self.client = None

    def init_client(self):
        self.client = weaviate.connect_to_custom(
            http_host=self.host,
            http_port=self.port,
            http_secure=False,
            grpc_host=self.host,
            grpc_port=50051,
            grpc_secure=False,
            skip_init_checks=True
        )

    def create_collection(self, dimension: int, config: Dict[str, Any] = None):
        config = config or {}
        m = config.get('m', 64)
        ef_construct = config.get('ef_construction', 128)
        
        if self.client.collections.exists(self.collection_name):
            self.client.collections.delete(self.collection_name)

        self.client.collections.create(
            name=self.collection_name,
            vectorizer_config=wvc.Configure.Vectorizer.none(),
            vector_index_config=wvc.Configure.VectorIndex.hnsw(
                max_connections=m,
                ef_construction=ef_construct,
                distance_metric=wvc.VectorDistances.COSINE
            ),
            properties=[
                wvc.Property(name="idx", data_type=wvc.DataType.INT)
            ]
        )

    def insert(self, vectors: List[List[float]], ids: List[int] = None):
        if ids is None:
            ids = list(range(len(vectors)))
            
        collection = self.client.collections.get(self.collection_name)
        
        data_objs = []
        for i, vec in zip(ids, vectors):
            data_objs.append(
                wvc.DataObject(
                    properties={"idx": i},
                    vector=vec
                )
            )
            
        collection.data.insert_many(data_objs)

    def refresh(self):
        # Weaviate is eventually consistent but for HNSW it's usually fast.
        # No explicit flush in API, but we can verify counts if needed.
        pass

    def search(self, query: List[float], k: int, search_params: Dict[str, Any] = None) -> List[int]:
        search_params = search_params or {}
        # ef can be set in search? V4 api might not expose it easily in query time 
        # without updating config, but let's check. 
        # Actually ef is an index config, but can be overridden?
        # In benchmark we might have to ignore dynamic ef for Weaviate if not supported per-query.
        # But we can update the collection config before search phase.
        
        collection = self.client.collections.get(self.collection_name)
        response = collection.query.near_vector(
            near_vector=query,
            limit=k,
            return_properties=["idx"]
        )
        
        return [o.properties["idx"] for o in response.objects]

    def clean(self):
        if self.client and self.client.collections.exists(self.collection_name):
            self.client.collections.delete(self.collection_name)
        if self.client:
            self.client.close()
