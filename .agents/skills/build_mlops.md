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


## Referência para o projeto existente

Este arquivo descreve o processo de autoria do Antigravity; seus objetivos não são evidências de implementação ou homologação. Para operar a aplicação existente, consulte o [índice atual da documentação](../../docs/README.md). Para mudanças, confira primeiro o [estado de atendimento dos requisitos](../../production_artifacts/Technical_Specification.md). O workflow `/startcycle` continua reservado à sua invocação explícita, com os gates definidos naquele fluxo.

O snapshot local de métricas é produzido automaticamente e servido por `metricas-treino`; é cache operacional, não substitui o histórico MLflow. A API e o treino usam jobs distintos no Prometheus. Não apresentar dados simulados de drift como observações reais.
