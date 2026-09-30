# Skill — Build MLOps

## MLflow
Usar Observer Pattern. Componentes de domínio emitem eventos; `ObservadorMlflow` transforma eventos em tracking.

## Registry e serving
- registrar modelo final no Model Registry;
- trabalhar com Champion/Challenger;
- servir por MLflow Model Serving;
- incluir assinatura, input example, pipeline e preprocessadores.

## Observabilidade
Provisionar Prometheus, Grafana, Loki e Alloy via Docker Compose. Criar dashboards para pipeline, Nested CV, tuning, treino, estatística, drift, EDA, serving, MLflow, staging, GLOBAL/ZONA/BAIRRO e infraestrutura.

## Persistência
Resultados importantes em MLflow/backend/object storage/staging, não em armazenamento manual local.
