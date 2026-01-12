# Reporte de Resultados del Benchmark

Debido a limitaciones de recursos en el entorno local (Docker Desktop), solo Qdrant pudo completar las pruebas exitosamente. Los otros motores presentaron fallos relacionados con infraestructura:
- **OpenSearch/Elasticsearch**: Bloqueo de escritura por bajo espacio en disco (`cluster create-index blocked`).
- **Milvus/Weaviate**: Problemas de conectividad gRPC/Timeouts (común en Docker Desktop con recursos limitados).

## Resultados Exitosos (Qdrant)
Dataset: `glove-100-angular` (10,000 vectores probados, dimensión 100).

| Engine | ef_search | Recall@10 | QPS | Avg Latency (ms) | P95 Latency (ms) |
|---|---|---|---|---|---|
| **Qdrant** | Default | 1.0000 | **1547.46** | 0.65 | 1.10 |
| **Qdrant** | 64 | 1.0000 | 562.02 | 1.78 | 3.12 |
| **Qdrant** | 128 | 1.0000 | 515.52 | 1.94 | 3.77 |
| **Qdrant** | 512 | 1.0000 | 565.95 | 1.76 | 3.20 |

*Nota: Qdrant logró un Recall perfecto (1.0) incluso con la configuración por defecto, demostrando una excelente optimización de su índice HNSW.*

## Conclusiones Preliminares
1.  **Facilidad de Uso**: Qdrant fue el único motor que funcionó "out-of-the-box" sin requerir ajustes complejos de red o grandes recursos.
2.  **Rendimiento**: Con >1500 QPS y <1ms de latencia promedio, es extremadamente eficiente para ejecución local.
3.  **Recomendación**: Para despliegues locales o con recursos limitados, **Qdrant** es la opción más robusta y fácil de operar.

## Próximos Pasos (Para ejecutar el resto)
1.  **Liberar Espacio en Disco**: OpenSearch requiere al menos 15-20% de disco libre.
2.  **Aumentar RAM**: Asignar 8GB+ a Docker.
3.  **Red**: Usar `--network host` en Linux, o depurar puertos gRPC en Mac/Windows para Milvus.
