# Índice da documentação

A documentação descreve o código atual. Exemplos numéricos não são benchmarks permanentes. YAMLs, código e resultados de cada execução são as fontes para parâmetros e métricas efetivos.

## Uso e operação

- [✨ Documentação Interativa em HTML](index.html).
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

## Requisitos e regras

- Especificações: [pipeline](../specs/01_pipeline_ml.md), [modelos](../specs/02_modelos_tuning.md), [dados e negócio](../specs/03_dados_negocio.md), [observabilidade](../specs/04_observabilidade.md).
- Regras: [Python](../rules/01_python_rules.md), [controle de fluxo](../rules/02_no_if_rules.md), [pandas](../rules/03_pandas_rules.md), [machine learning](../rules/04_ml_rules.md).
- Processo de autoria: [papéis](../.agents/agents.md) e [workflow startcycle](../.agents/workflows/startcycle.md). Esses arquivos definem um processo de desenvolvimento, não comprovam que todos os seus gates tenham sido satisfeitos.

## Convenções

**Implementado** descreve código presente; **pendente** descreve requisito ainda não atendido; **validado** exige evidência com escopo explícito. A atualização de um documento não altera modelos, infraestrutura nem configurações por si só. Consulte os resultados dos testes antes de afirmar aprovação global.
