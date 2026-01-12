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

---

## ☁️ Instrucciones para Ejecución Remota (EC2 / Cloud)

Para ejecutar este benchmark en una instancia de nube (ej. AWS EC2 `t3.large` o superior), sigue estos pasos optimizados para recursos limitados:

### 1. Preparar la Instancia
Conéctate a tu servidor y asegúrate de tener Docker y Git instalados.

```bash
# Actualizar y instalar dependencias básicas
sudo apt update && sudo apt install -y git docker.io docker-compose python3-venv

# Configurar permisos de Docker (evita usar sudo para docker)
sudo usermod -aG docker $USER
newgrp docker
```

### 2. Clonar el Repositorio
```bash
git clone https://github.com/eudesz/vector_database_benchmark.git
cd vector_database_benchmark
```

### 3. Configurar Entorno Python
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Configurar Swap (Opcional pero Recomendado)
Si usas una instancia con **8GB RAM o menos** (como `t3.large`), es crítico añadir Swap para evitar que el OOM Killer mate los procesos de Java (Elasticsearch/Milvus).

```bash
# Crear 4GB de swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 5. Configurar Memoria Virtual (Para Elastic/OpenSearch)
Elasticsearch requiere un límite alto de mapas de memoria virtual.
```bash
sudo sysctl -w vm.max_map_count=262144
# Para hacerlo permanente:
echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf
```

### 6. Ejecutar Benchmark Secuencial
Para evitar saturar la RAM, usa el script `run_local_sequential.sh`. Este script levanta un motor, ejecuta las pruebas, lo apaga y limpia antes de pasar al siguiente.

```bash
chmod +x run_local_sequential.sh
./run_local_sequential.sh
```

### 7. Ver Resultados
Al finalizar, los resultados estarán en:
```bash
cat benchmark_results.csv
```
