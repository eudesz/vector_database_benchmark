# Vector Database Benchmark

This project benchmarks 5 open-source vector databases:
1. **Qdrant**
2. **Milvus**
3. **Weaviate**
4. **OpenSearch**
5. **Elasticsearch**

## Prerequisites
- Docker & Docker Compose
- Python 3.10+
- 8GB+ RAM recommended (16GB+ for full dataset tests)

## Setup

1. **Start the Infrastructure**
   ```bash
   docker-compose up -d
   ```
   Wait for all services to be healthy. Milvus and Elasticsearch take the longest to start.

2. **Install Dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Running the Benchmark

The `benchmark.py` script handles data downloading, ingestion, and testing.

### Quick Smoke Test (Limit 1000 vectors)
Run a quick test on Qdrant to verify setup:
```bash
python benchmark.py --engines qdrant --limit 1000
```

### Full Benchmark (All Engines)
```bash
python benchmark.py --engines qdrant milvus weaviate opensearch elasticsearch
```

### Tuning Parameters
The script automatically tests multiple `ef_search` values to plot the Precision/Recall vs Latency curve.
See `TUNING_GUIDE.md` for details on how the engines are configured.

## Results
Results are printed to the console and saved to `benchmark_results.csv`.
