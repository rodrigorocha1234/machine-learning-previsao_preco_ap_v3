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


## Referência para o projeto existente

Este arquivo descreve o processo de autoria do Antigravity; seus objetivos não são evidências de implementação ou homologação. Para operar a aplicação existente, consulte o [índice atual da documentação](../../docs/README.md). Para mudanças, confira primeiro o [estado de atendimento dos requisitos](../../production_artifacts/Technical_Specification.md). O workflow `/startcycle` continua reservado à sua invocação explícita, com os gates definidos naquele fluxo.
