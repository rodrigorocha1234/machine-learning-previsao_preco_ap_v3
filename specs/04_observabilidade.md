# Spec 04 — Observabilidade e serving

## Requisitos

MLflow Tracking e Registry, backend DB, object storage S3 compatível, Prometheus, Grafana, Loki, Alloy e staging. Serving principal pela aplicação nativa MLflow. Separar resultados de treino/holdout das observações reais de produção.

## Implementação

- `servidor_inferencia` resolve `champion`, fixa a versão e inicializa o scoring nativo MLflow com um adaptador ASGI de telemetria.
- Job `mlflow_serving`: HTTP, lotes, problemas de entrada, distribuições de previsões por localidade, cobertura de referências e versão carregada. Um worker.
- Job `ml_service`: snapshot atômico do pipeline em `observabilidade_data/treino.prom`, servido continuamente por `metricas-treino:8000`.
- Grafana: painéis de operação, previsões, treino, holdout e seis comparações de média/desvio padrão por modelo. Barras usam labels diretamente, sem união por coluna inexistente.
- Média no período calculada por soma/contagem de imóveis. Medianas e percentis de serving são aproximações por histogramas.

## Pendências

Drift real de produção exige baseline versionado e janela de observações. Erros de previsão em produção exigem preço real de venda associado à previsão. O script de drift demonstrativo usa simulações e não é fonte de produção. O diagnóstico da etapa 9 compara a base consigo mesma.

Não há garantia de que todos os dashboards e alertas imaginados no desenho inicial tenham fonte implementada. [Catálogo e diagnóstico](../docs/observabilidade.md) e [operação](../production_artifacts/Deployment.md).
