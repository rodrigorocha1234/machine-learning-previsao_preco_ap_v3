---
description: Executa o ciclo multiagente completo do projeto ML imobiliário
---

Quando o usuário executar `/startcycle <objetivo>`, seguir estritamente `.agents/agents.md`, `.agents/skills/`, `specs/` e `rules/`.

## Sequência
1. @pm executa `write_specs.md` e atualiza `production_artifacts/Technical_Specification.md`.
2. PAUSAR no Approval Gate. Somente a palavra explícita `Approved` libera implementação.
3. @ml_architect executa `design_architecture.md`.
4. @data_engineer executa `build_data_layer.md`.
5. @ml_engineer executa `build_ml_pipeline.md`.
6. @mlops executa `build_mlops.md`.
7. @qa executa `audit_project.md` e corrige violações bloqueantes.
8. @devops executa `deploy_project.md`.
9. @qa executa validação final e grava `production_artifacts/Final_Audit.md`.

## Rework loop
Feedback do usuário ou comentários em artefatos reabrem a etapa responsável. Uma alteração arquitetural reabre @pm e @ml_architect. Alteração de regra estatística reabre @ml_architect e @ml_engineer. Alteração de infraestrutura reabre @mlops e @devops.


## Referência para o projeto existente

Este arquivo descreve o processo de autoria do Antigravity; seus objetivos não são evidências de implementação ou homologação. Para operar a aplicação existente, consulte o [índice atual da documentação](../../docs/README.md). Para mudanças, confira primeiro o [estado de atendimento dos requisitos](../../production_artifacts/Technical_Specification.md). O workflow `/startcycle` continua reservado à sua invocação explícita, com os gates definidos naquele fluxo.
