#!/bin/bash

# Activar entorno virtual si existe
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Archivo de resultados
RESULTS_FILE="benchmark_results.csv"

echo "=== INICIANDO BENCHMARK SECUENCIAL (Uno a Uno) ==="
echo "Esto optimizará el uso de RAM y CPU para evitar errores."

# Función para limpiar y apagar todo
cleanup() {
    echo "🧹 Limpiando contenedores..."
    docker-compose down
    sleep 5
}

# 1. QDRANT
cleanup
echo "🚀 [1/5] Iniciando Qdrant..."
docker-compose up -d qdrant
echo "Esperando 10s para arranque..."
sleep 10
python benchmark.py --engines qdrant --limit 10000

# 2. WEAVIATE
cleanup
echo "🚀 [2/5] Iniciando Weaviate..."
docker-compose up -d weaviate
echo "Esperando 15s para arranque..."
sleep 15
python benchmark.py --engines weaviate --limit 10000

# 3. MILVUS (Standalone)
cleanup
echo "🚀 [3/5] Iniciando Milvus (Standalone)..."
# Milvus requiere etcd y minio
docker-compose up -d etcd minio milvus
echo "Esperando 45s para arranque de Milvus..."
sleep 45
python benchmark.py --engines milvus --limit 10000

# 4. OPENSEARCH
cleanup
echo "🚀 [4/5] Iniciando OpenSearch..."
docker-compose up -d opensearch
echo "Esperando 30s para arranque..."
sleep 30
python benchmark.py --engines opensearch --limit 10000

# 5. ELASTICSEARCH
cleanup
echo "🚀 [5/5] Iniciando Elasticsearch..."
# Asegurar permisos para elasticsearch data si es necesario
# chmod 777 ./data/elasticsearch_data
docker-compose up -d elasticsearch
echo "Esperando 60s para arranque de Elasticsearch..."
sleep 60
# Verificar salud antes de lanzar
curl -X GET "localhost:9201/_cluster/health?wait_for_status=yellow&timeout=50s"
python benchmark.py --engines elasticsearch --limit 10000

cleanup
echo "✅ Benchmark completado. Resultados en $RESULTS_FILE"
cat $RESULTS_FILE
