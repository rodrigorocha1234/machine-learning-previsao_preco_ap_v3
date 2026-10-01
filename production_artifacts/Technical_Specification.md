# Especificação técnica e estado de implementação

## Objetivo

Estimar `Valor_da_Venda` de apartamentos da base de Ribeirão Preto/SP e apresentar referências globais, por zona e por bairro. A previsão é uma estimativa do modelo; as faixas e descontos são cálculos determinísticos, sem garantia de preço de venda ou de compra segura.

## Contratos

Entrada do treino: `dados/bairro_final_v3_engineered.xlsx`. A etapa de carga remove identificadores e colunas derivadas do preço. Entrada da API: seis atributos definidos no [contrato de integração](../docs/exemplo_chamada_api_mlflow.md). A saída atual possui 40 campos, incluindo localização, estimativas individuais, médias do lote e referências aprendidas na base de desenvolvimento.

## Etapas do fluxo

| Etapas | Implementação |
| --- | --- |
| 1–5 | Configuração, validação, carga do Excel, validação dos dados e staging |
| 6–7 | Split e bloqueio do holdout na etapa 6; etapa 7 é marcador |
| 8–9 | EDA no desenvolvimento e diagnóstico de drift sem comparação temporal real |
| 10–12 | Nested CV, testes estatísticos e seleção |
| 13–15 | Tuning final do líder, treino e flag de congelamento |
| 16–18 | Verificação da flag, liberação/avaliação do holdout na etapa 17 e regras de negócio |
| 19–20 | Registro/promoção MLflow na etapa 19; etapa 20 não inicia processos |

`FluxoPrincipal` executa as 20 etapas com observador MLflow. `scripts.recalcular_metricas` executa 1–18 e registra avaliações, sem registrar/promover um novo modelo.

## Modelagem

A configuração inicial usa holdout de 20%, RepeatedKFold externo 5 × 3 e KFold interno de 5 splits. Os modelos compartilham as mesmas divisões externas. GridSearchCV/RandomizedSearchCV ajustam pipelines completos; `nenhum` ajusta diretamente o pipeline. As buscas usam `n_jobs=None`; paralelismo do estimador é configuração distinta.

Cada modelo produz média, mediana e desvio padrão descritivo (`ddof=0`) de RMSE, MAE, MSE, R², RMSE relativo e MAPE. A dispersão mede variação entre divisões, não sensibilidade causal aos atributos nem intervalo de confiança. O ranking implementado utiliza RMSE/Friedman; a chave configurável de métrica principal não implica suporte integral a outros critérios em todos os componentes.

## Matriz de atendimento

| Requisito | Estado atual |
| --- | --- |
| Pré-processamento dentro da CV | Implementado |
| Holdout isolado do ajuste | Controle lógico por instância, com limitações descritas na arquitetura |
| Dez famílias de estimadores | Fábrica disponível; três ativas no YAML atual |
| Média, mediana e desvio entre folds | Implementado para seis métricas |
| Runs pai/filho e melhores parâmetros | Implementado para os resultados externos |
| Todo o histórico de tuning interno/final e YAML versionado | Persistência completa pendente; o resultado da busca existe em memória |
| Ensemble de votação | Pendente; a seleção lista componentes, mas treina apenas o líder |
| Fallback por suficiência em toda inferência | Parcial; API vetorizada usa presença da localidade |
| Descontos configuráveis por YAML | Pendente; percentuais atuais estão no código |
| Fontes substituíveis amplas | Excel/CSV/Parquet disponíveis; fluxo usa Excel explicitamente |
| Métricas contínuas de treino | Snapshot atômico e exportador persistente implementados |
| Métricas reais do serving | HTTP, entradas, distribuições e versão carregada implementados |
| Drift real e erro com preço de venda observado | Fontes e integração pendentes |
| Homologação completa das regras de engenharia | Não demonstrada |

## Observabilidade e critérios verificáveis

Os painéis de CV devem receber seis métricas por modelo e apresentar média/desvio com identificação explícita. O job `ml_service` deve continuar disponível após o fim do treinamento e reinício do exportador. Ausentes/outliers devem publicar zero quando medidos como zero. O job `mlflow_serving` deve contar requisições e imóveis separadamente, preservar as respostas da API e identificar a versão carregada.

A aceitação desses comportamentos não substitui uma auditoria global de segurança, estatística, qualidade de dados ou prontidão operacional. [QA](QA_Report.md) registra o escopo das evidências; [Final Audit](Final_Audit.md) lista pendências.
