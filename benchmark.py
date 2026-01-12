import argparse
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset import Dataset
from engines.qdrant_engine import QdrantEngine
from engines.milvus_engine import MilvusEngine
from engines.weaviate_engine import WeaviateEngine
from engines.opensearch_engine import OpenSearchEngine
from engines.elasticsearch_engine import ElasticsearchEngine

ENGINES = {
    "qdrant": QdrantEngine,
    "milvus": MilvusEngine,
    "weaviate": WeaviateEngine,
    "opensearch": OpenSearchEngine,
    "elasticsearch": ElasticsearchEngine,
}

def calculate_recall(results, ground_truth):
    """
    results: List of lists of IDs (predicted)
    ground_truth: List of lists of IDs (actual)
    """
    total_recall = 0
    k = len(results[0])
    
    for pred, actual in zip(results, ground_truth):
        intersection = len(set(pred) & set(actual))
        total_recall += intersection / k
        
    return total_recall / len(results)

def run_benchmark(engine_name, dataset_name, limit=None, ef_search_list=[None]):
    print(f"\n=== Benchmarking {engine_name.upper()} on {dataset_name} ===")
    
    # Load Dataset
    ds = Dataset(dataset_name)
    ds.load()
    
    train_vectors = ds.train_vectors
    test_vectors = ds.test_vectors
    ground_truth = ds.get_ground_truth(k=10) # Testing Top-10
    
    if limit:
        train_vectors = train_vectors[:limit]
        print(f"Limiting dataset to {limit} vectors.")

    # Init Engine
    engine_cls = ENGINES[engine_name]
    engine = engine_cls(collection_name="benchmark_test")
    
    try:
        engine.init_client()
        # 1. Create Collection & Index
        print("Creating collection...")
        start_time = time.time()
        engine.create_collection(ds.dimension, config={"m": 16, "ef_construction": 128})
        print(f"Collection created in {time.time() - start_time:.2f}s")

        # 2. Ingestion
        print("Ingesting vectors...")
        start_time = time.time()
        engine.insert(train_vectors.tolist())
        ingest_time = time.time() - start_time
        print(f"Ingestion finished in {ingest_time:.2f}s ({len(train_vectors)/ingest_time:.0f} vecs/s)")
        
        # 3. Refresh/Index
        print("Refreshing/Indexing...")
        start_time = time.time()
        engine.refresh()
        print(f"Refresh finished in {time.time() - start_time:.2f}s")
        
        # 4. Search Benchmark
        results_data = []
        
        for ef in ef_search_list:
            print(f"\n--- Testing Search (ef_search={ef if ef else 'default'}) ---")
            search_params = {"ef_search": ef} if ef else {}
            
            latencies = []
            predictions = []
            
            start_test = time.time()
            for query in tqdm(test_vectors):
                t0 = time.time()
                ids = engine.search(query.tolist(), k=10, search_params=search_params)
                latencies.append(time.time() - t0)
                predictions.append(ids)
            
            total_time = time.time() - start_test
            
            # Metrics
            recall = calculate_recall(predictions, ground_truth)
            avg_latency = np.mean(latencies) * 1000 # ms
            p95_latency = np.percentile(latencies, 95) * 1000
            p99_latency = np.percentile(latencies, 99) * 1000
            qps = len(test_vectors) / total_time
            
            print(f"Results for {engine_name} (ef={ef}):")
            print(f"Recall@10: {recall:.4f}")
            print(f"QPS: {qps:.2f}")
            print(f"Avg Latency: {avg_latency:.2f}ms")
            print(f"P95 Latency: {p95_latency:.2f}ms")
            
            results_data.append({
                "engine": engine_name,
                "dataset": dataset_name,
                "ef_search": ef,
                "recall": recall,
                "qps": qps,
                "avg_latency": avg_latency,
                "p95_latency": p95_latency,
                "ingest_time": ingest_time
            })
            
        return results_data

    except Exception as e:
        print(f"Error running benchmark for {engine_name}: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        print("Cleaning up...")
        engine.clean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector DB Benchmark")
    parser.add_argument("--engines", nargs="+", default=["qdrant"], help="Engines to test")
    parser.add_argument("--dataset", default="glove-100-angular", help="Dataset name")
    parser.add_argument("--limit", type=int, help="Limit number of vectors for quick test")
    
    args = parser.parse_args()
    
    all_results = []
    
    # Define search configurations to test (Tuning Phase)
    # We test default (None), then some values to see trade-off
    ef_configs = [None, 64, 128, 256, 512]
    
    for engine in args.engines:
        if engine not in ENGINES:
            print(f"Unknown engine: {engine}")
            continue
            
        res = run_benchmark(engine, args.dataset, args.limit, ef_configs)
        all_results.extend(res)
        
    # Save Summary
    if all_results:
        df = pd.DataFrame(all_results)
        print("\n=== FINAL REPORT ===")
        print(df.to_markdown(index=False))
        
        # Append to csv if exists, else write new
        import os
        header = not os.path.exists("benchmark_results.csv")
        df.to_csv("benchmark_results.csv", mode='a', header=header, index=False)
