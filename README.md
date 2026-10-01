# Previsão de preços de apartamentos — Ribeirão Preto/SP

Projeto Python de regressão imobiliária com validação cruzada aninhada, MLflow Model Registry e Serving, regras de referência geográfica e monitoramento no Grafana.

## Comece por aqui

| Necessidade | Documento |
| --- | --- |
| Preparar ambiente e executar serviços | [Implantação e operação](production_artifacts/Deployment.md) |
| Chamar a API e interpretar seus 40 campos | [Contrato da API](docs/exemplo_chamada_api_mlflow.md) |
| Entender treino, tuning e desvio padrão | [Validação cruzada](docs/validacao_cruzada_e_tuning.md) |
| Usar Grafana e resolver painéis sem dados | [Observabilidade](docs/observabilidade.md) |
| Entender componentes e limitações | [Arquitetura](production_artifacts/Architecture.md) |
| Consultar requisitos e estado de implementação | [Especificação técnica](production_artifacts/Technical_Specification.md) |
| Conferir evidências de testes | [Relatório de qualidade](production_artifacts/QA_Report.md) |
| Navegar por todos os documentos | [Índice de documentação](docs/README.md) |

## Fluxos de execução

Execute os comandos na raiz, com o ambiente Python e os serviços configurados conforme o guia de implantação.

```bash
# Recalcula treino, CV e holdout; salva métricas, sem promover outro modelo
.venv/bin/python -m scripts.recalcular_metricas

# Fluxo completo: também registra o modelo e atualiza o alias champion
.venv/bin/python -m app_build.orquestracao_pipeline.fluxo_principal

# Exemplo de consumo da API
.venv/bin/python chamada.py
```

A fonte atual é `dados/bairro_final_v3_engineered.xlsx`. O alvo é `Valor_da_Venda`; as seis entradas da API são Bairro, Zona, Quartos, Banheiros, Vagas_Garagem e Metragem.

## Comportamento atual

- Holdout de 20%; desenvolvimento de 80%, com semente 42. A separação usa embaralhamento, sem estratificação efetiva.
- CV externa: 5 folds × 3 repetições; CV interna: 5 folds. Pré-processamento ajustado dentro do pipeline de cada busca.
- Ridge, árvore de decisão e Random Forest ativos no YAML atual; dez famílias de estimadores disponíveis.
- Médias, medianas e desvios padrão (`ddof=0`) das seis métricas entre folds externos.
- Com `selecao_modelos.votacao: true`, o `VotingRegressor` combina por média os top K modelos, cada um com tuning e pré-processamento próprios. Com `false`, treina o vencedor individual.
- Preços previstos por zona/bairro na API são médias **do lote enviado**. No Grafana, as médias de serving resumem os imóveis atendidos no período selecionado.
- Métricas de treino são persistidas em `observabilidade_data/treino.prom` e servidas continuamente pelo contêiner `metricas-treino`.
- O serving resolve `champion` ao iniciar e fixa a versão carregada. Uma nova avaliação sem promoção não altera a API.

## Estrutura

`app_build/` contém a aplicação; `configs/`, configuração; `scripts/`, utilitários operacionais; `tests/`, testes; `config_ob/`, observabilidade; `docs/`, guias; `production_artifacts/`, descrição técnica e evidências; `specs/` e `rules/`, requisitos e regras de desenvolvimento.

`.agents/` conserva o fluxo de autoria multiagente do Antigravity. `/startcycle` é um procedimento de desenvolvimento explicitamente invocado, não o comando para executar a aplicação existente.

## Limites relevantes

O holdout possui proteção lógica em memória, não criptografia. As regras de suficiência amostral não estão integradas ao enriquecimento vetorizado da API. Drift real de produção e erros associados ao preço real de venda ainda dependem de fontes adicionais. Não há uma certificação atual de conformidade integral ou de prontidão irrestrita para produção; veja [a revisão final](production_artifacts/Final_Audit.md).
