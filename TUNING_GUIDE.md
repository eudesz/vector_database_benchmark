# Guía de Fine-Tuning para Motores Vectoriales

Este documento detalla los parámetros críticos de configuración para optimizar el rendimiento (QPS) y la precisión (Recall) en los motores evaluados.

## Conceptos Clave (HNSW)
La mayoría de estos motores utilizan **HNSW** (Hierarchical Navigable Small World) como algoritmo de indexación.
- **M (Max Links)**: Número máximo de conexiones por nodo.
  - *Alto*: Mejor Recall, índice más grande, inserción más lenta.
  - *Bajo*: Menor memoria, inserción más rápida.
- **efConstruction**: Profundidad de búsqueda durante la construcción del índice.
  - *Alto*: Mejor calidad de índice (Recall), inserción mucho más lenta.
  - *Bajo*: Inserción rápida, menor calidad.
- **ef (Search) / num_candidates**: Profundidad de búsqueda durante la consulta.
  - *Alto*: Mejor Recall, mayor Latencia (menos QPS).
  - *Bajo*: Búsqueda rápida, menor precisión.

---

## 1. Qdrant
**Configuración de Colección:**
```json
{
  "vectors": {
    "size": 128,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,            // Default: 16. Rango: 8-64
    "ef_construct": 100 // Default: 100. Rango: 40-400
  }
}
```
**Configuración de Búsqueda:**
```json
{
  "params": {
    "hnsw_ef": 128      // Ajustar dinámicamente. Default suele ser bajo.
  }
}
```

## 2. Milvus
**Creación de Índice:**
```python
index_params = {
  "metric_type": "L2",
  "index_type": "HNSW",
  "params": {
    "M": 16,              // Rango: 4-64
    "efConstruction": 200 // Rango: 8-512
  }
}
```
**Parámetros de Búsqueda:**
```python
search_params = {
  "metric_type": "L2",
  "params": {
    "ef": 64              // Crítico para Latencia vs Recall
  }
}
```

## 3. Weaviate
**Schema (Class Definition):**
```json
{
  "vectorIndexType": "hnsw",
  "vectorIndexConfig": {
    "maxConnections": 64,   // Equivale a M. Default: 64
    "efConstruction": 128,  // Default: 128
    "ef": -1                // -1 = dinámico/auto. Se puede fijar para benchmarks.
  }
}
```

## 4. OpenSearch (k-NN Plugin)
**Mapping:**
```json
{
  "properties": {
    "my_vector": {
      "type": "knn_vector",
      "dimension": 128,
      "method": {
        "name": "hnsw",
        "engine": "lucene", // O "nmslib" / "faiss" (deprecated en versiones nuevas default a lucene)
        "parameters": {
          "m": 16,
          "ef_construction": 100
        }
      }
    }
  }
}
```
**Search:**
```json
{
  "size": 10,
  "query": {
    "knn": {
      "my_vector": {
        "vector": [...],
        "k": 10
      }
    }
  }
}
```
*Nota*: OpenSearch también permite configurar `knn.algo_param.ef_search` en la configuración del índice.

## 5. Elasticsearch
**Mapping:**
```json
{
  "properties": {
    "my_vector": {
      "type": "dense_vector",
      "dims": 128,
      "index": true,
      "similarity": "cosine",
      "index_options": {
        "type": "hnsw",
        "m": 16,
        "ef_construction": 100
      }
    }
  }
}
```
**Search:**
```json
{
  "knn": {
    "field": "my_vector",
    "query_vector": [...],
    "k": 10,
    "num_candidates": 100 // Similar a ef_search. Debe ser >= k.
  }
}
```

## Estrategia de Benchmark
1. **Baseline**: Usar defaults de cada motor.
2. **Tuned**:
   - Fijar `M=32` y `efConstruction=200` en todos para nivelar el terreno de juego en indexación.
   - Variar `ef_search` / `num_candidates` durante las pruebas de consulta (ej. 10, 50, 100, 200) para trazar la curva Latencia/Recall.
