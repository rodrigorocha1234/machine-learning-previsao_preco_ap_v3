# Skill — Deploy Project

## Objetivo
Empacotar e executar o projeto aprovado.

## Etapas
- validar dependências;
- construir imagens;
- executar Docker Compose;
- validar healthchecks/readiness;
- validar conexão aplicação → MLflow → object storage;
- validar Prometheus/Grafana/Loki/Alloy;
- validar endpoint de MLflow Serving;
- registrar comandos reproduzíveis em `production_artifacts/Deployment.md`.
