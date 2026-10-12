# Índice da documentação

A documentação descreve o código atual. Exemplos numéricos não são benchmarks permanentes. YAMLs, código e resultados de cada execução são as fontes para parâmetros e métricas efetivos.

## Uso e operação

- [✨ Documentação Interativa em HTML](index.html).
- [Fluxo Detalhado das 20 Etapas do Pipeline](fluxo_das_etapas.md).
- [Visão geral](../README.md).
- [Guia passo a passo de como executar o projeto](como_executar_o_projeto.md).
- [Instalação, treinamento, publicação e diagnóstico](../production_artifacts/Deployment.md).
- [Contrato da API](exemplo_chamada_api_mlflow.md).
- [Validação cruzada, VotingRegressor e sensibilidade às divisões dos dados](validacao_cruzada_e_tuning.md).
- [Grafana, Prometheus, persistência e catálogo de métricas](observabilidade.md).

## Engenharia e evidências

- [Especificação e estado de implementação](../production_artifacts/Technical_Specification.md).
- [Arquitetura implementada](../production_artifacts/Architecture.md).
- [Decisões e limitações](../production_artifacts/Decision_Log.md).
- [Qualidade e validações](../production_artifacts/QA_Report.md).
- [Revisão final e pendências](../production_artifacts/Final_Audit.md).

## Requisitos, Agentes e Specs

- **Equipe Multiagente**: [Papéis dos 7 agentes](../.agents/agents.md) e [workflow /startcycle com Approval Gate](../.agents/workflows/startcycle.md).
- **Especificações Formais (Specs)**:
  - [Spec 01 — Pipeline de ML](../specs/01_pipeline_ml.md) (20 etapas, RepeatedKFold 5×3, cofre de holdout, Friedman/Nemenyi).
  - [Spec 02 — Modelos e Tuning](../specs/02_modelos_tuning.md) (10 famílias, tuning adaptativo, VotingRegressor).
  - [Spec 03 — Dados e Negócio](../specs/03_dados_negocio.md) (Target Valor_da_Venda, hierarquia em 3 níveis, suficiência 30/20, corredores seguros).
  - [Spec 04 — Observabilidade e Serving](../specs/04_observabilidade.md) (MLflow Serving com adaptador ASGI, 56 métricas Prometheus, persistência atômica, Grafana).
- **Regras Arquiteturais Inegociáveis**:
  - [Python 3.12 Estrito & Sem Any](../rules/01_python_rules.md).
  - [Política "No-If" em Regras de Domínio](../rules/02_no_if_rules.md) (padrões GoF: Strategy, State, Specification, Dispatch Tables).
  - [Vetorização Pura em Pandas](../rules/03_pandas_rules.md) (zero loops iterativos).
  - [Governança de ML & Anti-Leakage](../rules/04_ml_rules.md).

## Convenções

**Implementado** descreve código presente; **pendente** descreve requisito ainda não atendido; **validado** exige evidência com escopo explícito. A atualização de um documento não altera modelos, infraestrutura nem configurações por si só. Consulte os resultados dos testes antes de afirmar aprovação global.
