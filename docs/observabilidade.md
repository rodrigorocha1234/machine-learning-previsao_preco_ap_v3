# Observabilidade — Stack de Monitoramento

> **Projeto:** Previsão de Preços de Apartamentos — Ribeirão Preto/SP  
> **Stack:** Prometheus · Grafana · Loki · Grafana Alloy

---

## Sumário

1. [Arquitetura da Stack](#1-arquitetura-da-stack)
2. [Catálogo Completo de Métricas Prometheus](#2-catálogo-completo-de-métricas-prometheus)
3. [Onde Cada Métrica é Coletada](#3-onde-cada-métrica-é-coletada)
4. [Logs Estruturados — Loki](#4-logs-estruturados--loki)
5. [Dashboards Grafana](#5-dashboards-grafana)
6. [Alertas Recomendados](#6-alertas-recomendados)
7. [Arquivos de Referência](#7-arquivos-de-referência)

---

## 1. Arquitetura da Stack

```
┌──────────────────────────────────────────────────────────────────────┐
│                         PIPELINE PYTHON                              │
│                                                                      │
│  FluxoPrincipal                                                      │
│   ├── ServicoTelemetria ──→ ColetorPrometheus ──→ :8000/metrics      │
│   ├── EmissorLogs       ──→ stdout/stderr ──→ Alloy ──→ Loki         │
│   └── ExecutorEsteira   ──→ registra etapas em tempo real            │
│        ├── Etapa04ValidarDados   → qualidade_dados                   │
│        ├── Etapa09Drift          → drift_dados, drift_feature        │
│        ├── Etapa10NestedCv       → cv_rmse_fold, cv_resumo           │
│        ├── Etapa11Estatistica    → testes_estatisticos               │
│        ├── Etapa17AvaliacaoHoldout → holdout_zona, holdout_bairro    │
│        └── Etapa18RegrasNegocio  → metricas_negocio                  │
└──────────────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
  ┌─────────────┐                    ┌──────────────┐
  │  Prometheus │◄──scrape :8000─── │   :8000/     │
  │   :9090     │                    │   metrics    │
  └──────┬──────┘                    └──────────────┘
         │
         ▼
  ┌─────────────┐     ┌───────────┐
  │   Grafana   │◄────│   Loki    │
  │   :3000     │     │   :3100   │
  └─────────────┘     └───────────┘
```

---

## 2. Catálogo Completo de Métricas Prometheus

### 2.1 Inferência em Tempo Real

| Métrica | Tipo | Labels | Descrição |
| :--- | :---: | :--- | :--- |
| `apartamentos_predicoes_total` | Counter | `zona` | Número acumulado de predições realizadas |
| `apartamentos_latencia_predicao_seconds` | Histogram | — | Distribuição de latência das predições |
| `apartamentos_valor_medio_previsto` | Gauge | `zona` | Último valor médio previsto por zona |
| `apartamentos_predicao_valor_previsto` | Histogram | — | Distribuição dos valores previstos (buckets R$) |
| `apartamentos_predicao_residuo_medio` | Gauge | — | Viés médio do modelo (resíduo médio) |
| `apartamentos_predicao_residuo_std` | Gauge | — | Dispersão dos resíduos (desvio padrão) |
| `apartamentos_predicao_erro_percentual_p50` | Gauge | — | Mediana do erro percentual absoluto |
| `apartamentos_predicao_erro_percentual_p90` | Gauge | — | Percentil 90 do erro percentual absoluto |
| `apartamentos_predicao_erro_percentual_p95` | Gauge | — | Percentil 95 do erro percentual absoluto |

### 2.2 Performance do Modelo (Holdout Global)

| Métrica | Tipo | Labels | Descrição |
| :--- | :---: | :--- | :--- |
| `apartamentos_modelo_rmse` | Gauge | — | RMSE no conjunto de holdout (R$) |
| `apartamentos_modelo_mae` | Gauge | — | MAE no conjunto de holdout (R$) |
| `apartamentos_modelo_r2` | Gauge | — | R² no conjunto de holdout |
| `apartamentos_modelo_mape` | Gauge | — | MAPE no conjunto de holdout (%) |
| `apartamentos_modelo_amostras_treino` | Gauge | — | Total de amostras usadas no treino |
| `apartamentos_modelo_amostras_holdout` | Gauge | — | Total de amostras no holdout |
| `apartamentos_modelo_info` | Info | `versao`, `nome` | Versão e nome do modelo campeão |

### 2.3 Holdout Granular — Zona e Bairro ⭐ *novo*

| Métrica | Tipo | Labels | Descrição |
| :--- | :---: | :--- | :--- |
| `apartamentos_holdout_rmse_zona` | Gauge | `zona` | RMSE do holdout por zona territorial |
| `apartamentos_holdout_mae_zona` | Gauge | `zona` | MAE do holdout por zona territorial |
| `apartamentos_holdout_r2_zona` | Gauge | `zona` | R² do holdout por zona territorial |
| `apartamentos_holdout_mape_zona` | Gauge | `zona` | MAPE do holdout por zona (%) |
| `apartamentos_holdout_amostras_zona` | Gauge | `zona` | Amostras no holdout por zona |
| `apartamentos_holdout_rmse_bairro` | Gauge | `bairro`, `zona` | RMSE do holdout por bairro |
| `apartamentos_holdout_r2_bairro` | Gauge | `bairro`, `zona` | R² do holdout por bairro |

### 2.4 Validação Cruzada — Por Fold e Por Modelo ⭐ *novo*

| Métrica | Tipo | Labels | Descrição |
| :--- | :---: | :--- | :--- |
| `apartamentos_cv_rmse_fold` | Gauge | `modelo`, `fold` | RMSE de cada fold externo do Nested CV |
| `apartamentos_cv_r2_fold` | Gauge | `modelo`, `fold` | R² de cada fold externo do Nested CV |
| `apartamentos_cv_mae_fold` | Gauge | `modelo`, `fold` | MAE de cada fold externo do Nested CV |
| `apartamentos_cv_mape_fold` | Gauge | `modelo`, `fold` | MAPE de cada fold externo do Nested CV |
| `apartamentos_cv_rmse_medio` | Gauge | `modelo` | RMSE médio dos 15 folds externos |
| `apartamentos_cv_r2_medio` | Gauge | `modelo` | R² médio dos 15 folds externos |
| `apartamentos_cv_rmse_std` | Gauge | `modelo` | Desvio padrão do RMSE entre os folds |
| `apartamentos_cv_duracao_media_fold_segundos` | Gauge | `modelo` | Duração média por fold externo |

### 2.5 Testes Estatísticos

| Métrica | Tipo | Labels | Descrição |
| :--- | :---: | :--- | :--- |
| `apartamentos_estatistica_friedman_chi2` | Gauge | — | Estatística χ² do teste de Friedman |
| `apartamentos_estatistica_friedman_pvalor` | Gauge | — | P-valor do teste de Friedman |
| `apartamentos_estatistica_friedman_significativo` | Gauge | — | 1 se significativo (α<0.05), 0 caso contrário |
| `apartamentos_estatistica_nemenyi_cd` | Gauge | — | Diferença Crítica do teste de Nemenyi |
| `apartamentos_estatistica_rank_modelo` | Gauge | `modelo` | Rank médio de Friedman por modelo |
| `apartamentos_estatistica_nemenyi_diferenca_par` | Gauge | `modelo_a`, `modelo_b` | Diferença de ranks entre par de modelos |
| `apartamentos_estatistica_nemenyi_par_significativo` | Gauge | `modelo_a`, `modelo_b` | 1 se par é significativamente diferente |
| `apartamentos_estatistica_shapiro_pvalor` | Gauge | `modelo` | P-valor do Shapiro-Wilk por modelo |
| `apartamentos_estatistica_shapiro_normal` | Gauge | `modelo` | 1 se resíduos seguem distribuição normal |

### 2.6 Detecção de Drift

| Métrica | Tipo | Labels | Descrição |
| :--- | :---: | :--- | :--- |
| `apartamentos_drift_psi_predicoes` | Gauge | — | PSI médio das predições vs. referência |
| `apartamentos_drift_psi_area` | Gauge | — | PSI da feature Metragem |
| `apartamentos_drift_ks_area_stat` | Gauge | — | Estatística KS da feature Metragem |
| `apartamentos_drift_ks_area_pvalor` | Gauge | — | P-valor KS da feature Metragem |
| `apartamentos_drift_wasserstein_area` | Gauge | — | Distância Wasserstein normalizada (Metragem) |
| `apartamentos_drift_status_geral` | Gauge | — | 0=estável, 1=alerta, 2=crítico |
| `apartamentos_drift_psi_features` | Gauge | `feature` | PSI por feature individual |
| `apartamentos_drift_desvio_preco_m2_zona` | Gauge | `zona` | Desvio % do preço/m² por zona vs. global |

### 2.7 Qualidade dos Dados ⭐ *novo*

| Métrica | Tipo | Labels | Descrição |
| :--- | :---: | :--- | :--- |
| `apartamentos_dados_total_amostras` | Gauge | — | Total de imóveis no dataset bruto |
| `apartamentos_dados_missing_percentual` | Gauge | `coluna` | % de valores ausentes por coluna |
| `apartamentos_dados_outliers_percentual` | Gauge | `coluna` | % de outliers por coluna (método IQR) |
| `apartamentos_dados_media_alvo` | Gauge | — | Média do target (Valor_da_Venda) |
| `apartamentos_dados_mediana_alvo` | Gauge | — | Mediana do target |
| `apartamentos_dados_std_alvo` | Gauge | — | Desvio padrão do target |
| `apartamentos_dados_assimetria_alvo` | Gauge | — | Assimetria (skewness) do target |
| `apartamentos_dados_amostras_por_zona` | Gauge | `zona` | Imóveis por zona no dataset dev |
| `apartamentos_dados_amostras_por_bairro` | Gauge | `bairro` | Imóveis por bairro no dataset dev |

### 2.8 Métricas de Negócio Hierárquico

| Métrica | Tipo | Labels | Descrição |
| :--- | :---: | :--- | :--- |
| `apartamentos_negocio_preco_mediano_global` | Gauge | — | Preço mediano global (R$) |
| `apartamentos_negocio_preco_medio_global` | Gauge | — | Preço médio global (R$) |
| `apartamentos_negocio_preco_m2_mediano_global` | Gauge | — | Preço mediano por m² global |
| `apartamentos_negocio_total_imoveis` | Gauge | — | Total de imóveis no dataset |
| `apartamentos_negocio_preco_mediano_zona` | Gauge | `zona` | Preço mediano por zona (R$) |
| `apartamentos_negocio_preco_m2_mediano_zona` | Gauge | `zona` | Preço/m² mediano por zona |
| `apartamentos_negocio_total_amostras_zona` | Gauge | `zona` | Total de imóveis por zona |
| `apartamentos_negocio_preco_mediano_bairro` | Gauge | `bairro`, `zona` | Preço mediano por bairro (R$) |
| `apartamentos_negocio_total_amostras_bairro` | Gauge | `bairro`, `zona` | Total de imóveis por bairro |

### 2.9 Pipeline e Execução

| Métrica | Tipo | Labels | Descrição |
| :--- | :---: | :--- | :--- |
| `apartamentos_pipeline_etapa_ativa` | Gauge | — | Índice da etapa em execução (0=idle) |
| `apartamentos_pipeline_total_etapas` | Gauge | — | Total de etapas do pipeline |
| `apartamentos_pipeline_etapas_iniciadas_total` | Counter | — | Etapas iniciadas acumuladas |
| `apartamentos_pipeline_etapas_concluidas_total` | Counter | — | Etapas concluídas com sucesso |
| `apartamentos_pipeline_etapas_falhas_total` | Counter | — | Etapas que falharam |
| `apartamentos_pipeline_duracao_etapa_seconds` | Histogram | `etapa` | Distribuição de duração por etapa |
| `apartamentos_pipeline_duracao_total_segundos` | Gauge | — | Duração total do último pipeline (s) |
| `apartamentos_pipeline_ultimo_timestamp` | Gauge | — | Timestamp Unix da última execução |
| `apartamentos_pipeline_execucoes_total` | Counter | — | Total de execuções do pipeline |
| `apartamentos_pipeline_etapas_com_falha` | Gauge | — | Nº de etapas com falha na última execução |

### 2.10 Recursos do Sistema ⭐ *novo*

| Métrica | Tipo | Labels | Descrição |
| :--- | :---: | :--- | :--- |
| `apartamentos_sistema_cpu_uso_percentual` | Gauge | — | % de CPU consumido pelo processo |
| `apartamentos_sistema_memoria_uso_mb` | Gauge | — | RAM consumida pelo processo (MB) |
| `apartamentos_sistema_threads_ativas` | Gauge | — | Threads ativas no processo |

---

## 3. Onde Cada Métrica é Coletada

```
Etapa do Pipeline          Métricas Publicadas
─────────────────────────────────────────────────────────────────────
Etapa04ValidarDados     →  dados_total_amostras, dados_missing_*,
                           dados_outliers_*, dados_media/mediana/std/
                           assimetria_alvo, dados_amostras_por_zona/bairro,
                           sistema_*

Etapa09Drift            →  drift_psi_predicoes, drift_psi_area,
                           drift_ks_area_*, drift_wasserstein_area,
                           drift_status_geral, drift_psi_features{feature},
                           drift_desvio_preco_m2_zona{zona}

Etapa10NestedCv         →  cv_rmse_fold{modelo,fold}, cv_r2_fold,
  (por modelo)             cv_mae_fold, cv_mape_fold,
                           cv_rmse_medio{modelo}, cv_r2_medio,
                           cv_rmse_std, cv_duracao_media_fold_segundos,
                           sistema_*

Etapa11Estatistica      →  estatistica_friedman_chi2, friedman_pvalor,
  (via despachante)        friedman_significativo, nemenyi_cd,
                           rank_modelo{modelo}, nemenyi_diferenca_par,
                           nemenyi_par_significativo, shapiro_pvalor/normal

Etapa17Holdout          →  holdout_rmse/mae/r2/mape_zona{zona},
                           holdout_amostras_zona,
                           holdout_rmse/r2_bairro{bairro,zona},
                           predicao_residuo_medio/std,
                           predicao_erro_percentual_p50/p90/p95,
                           predicao_valor_previsto (histogram),
                           sistema_*

Etapa18RegrasNegocio    →  negocio_preco_mediano/medio_global,
  (via despachante)        negocio_preco_m2_mediano_global,
                           negocio_total_imoveis,
                           negocio_preco_mediano_zona{zona},
                           negocio_preco_m2_mediano_zona,
                           negocio_total_amostras_zona,
                           negocio_preco_mediano_bairro{bairro,zona},
                           negocio_total_amostras_bairro

ExecutorEsteira         →  pipeline_etapa_ativa, pipeline_total_etapas,
  (toda etapa)             pipeline_etapas_iniciadas/concluidas/falhas,
                           pipeline_duracao_etapa_seconds{etapa}

ExecutorEsteira         →  pipeline_duracao_total_segundos,
  (fim do pipeline)        pipeline_ultimo_timestamp,
                           pipeline_execucoes_total,
                           pipeline_etapas_com_falha

API Serving (:8080)     →  predicoes_total{zona}, latencia_predicao_seconds,
  (via EmpacotadorModelo)  valor_medio_previsto{zona}
```

---

## 4. Logs Estruturados — Loki

Todos os logs do pipeline são emitidos pelo `EmissorLogs` via `stdout/stderr` e coletados pelo **Grafana Alloy** → **Loki**.

### Padrão das mensagens

```
=== Iniciando Pipeline de Previsao de Precos de Apartamentos ===
>>> Executando etapa: 10_executar_nested_cv (Etapa 10 de 20)
--- Concluida etapa: 10_executar_nested_cv (Etapa 10 de 20) [109.19s]
>>> Executando etapa: 11_executar_estatistica (Etapa 11 de 20)
```

### Query LogQL úteis

```logql
# Todas as etapas concluídas com duração
{job="pipeline"} |= "Concluida etapa"

# Erros e avisos
{job="pipeline"} |= "Traceback" or "Error" or "Aviso"

# Apenas o modelo selecionado
{job="pipeline"} |= "selecao_modelo_final"

# Taxa de mensagens por minuto
rate({job="pipeline"}[1m])

# Filtrar por etapa específica
{job="pipeline"} |= "nested_cv"
```

---

## 5. Dashboards Grafana

O dashboard `dashboard_geral_imobiliario.json` organiza os painéis em seções:

| Seção | Painéis | Principais queries |
| :--- | :--- | :--- |
| **Visão Geral do Pipeline** | Status atual, progresso, duração por etapa | `apartamentos_pipeline_*` |
| **Logs de Andamento** | Log stream em tempo real | LogQL Loki |
| **Performance do Modelo** | RMSE/MAE/R²/MAPE global | `apartamentos_modelo_*` |
| **Validação Cruzada** | RMSE por fold e por modelo | `apartamentos_cv_rmse_fold` |
| **Holdout por Zona** | Mapa de calor zona × métrica | `apartamentos_holdout_*_zona` |
| **Holdout por Bairro** | Tabela RMSE/R² granular | `apartamentos_holdout_*_bairro` |
| **Testes Estatísticos** | Friedman, Nemenyi, Shapiro | `apartamentos_estatistica_*` |
| **Drift de Dados** | PSI por feature, KS, Wasserstein | `apartamentos_drift_*` |
| **Qualidade dos Dados** | Missing %, outliers %, distribuição | `apartamentos_dados_*` |
| **Predições em Produção** | Distribuição de valores, P50/P90/P95 erro | `apartamentos_predicao_*` |
| **Métricas de Negócio** | Preços medianos global/zona/bairro | `apartamentos_negocio_*` |
| **Recursos do Sistema** | CPU %, Memória MB, Threads | `apartamentos_sistema_*` |

### Exemplos de queries PromQL

```promql
# RMSE médio por modelo no CV
apartamentos_cv_rmse_medio

# Evolução do RMSE por fold do modelo campeão
apartamentos_cv_rmse_fold{modelo="ridge"}

# Diferença de RMSE entre zonas no holdout
apartamentos_holdout_rmse_zona

# Erro P95 das predições em produção
apartamentos_predicao_erro_percentual_p95

# Status de drift (0=ok, 1=alerta, 2=crítico)
apartamentos_drift_status_geral

# % outliers na feature mais crítica
apartamentos_dados_outliers_percentual{coluna="Valor_da_Venda"}

# Memória consumida durante o pipeline
apartamentos_sistema_memoria_uso_mb
```

---

## 6. Alertas Recomendados

| Alerta | Condição PromQL | Severidade |
| :--- | :--- | :---: |
| Drift crítico detectado | `apartamentos_drift_status_geral >= 2` | 🔴 Critical |
| PSI acima do limite de alerta | `apartamentos_drift_psi_predicoes > 0.10` | 🟡 Warning |
| RMSE do holdout degradou >10% | `apartamentos_modelo_rmse > RMSE_baseline * 1.10` | 🟡 Warning |
| Erro P95 > 30% | `apartamentos_predicao_erro_percentual_p95 > 30` | 🟡 Warning |
| Etapas com falha na execução | `apartamentos_pipeline_etapas_com_falha > 0` | 🔴 Critical |
| Latência de predição alta | `histogram_quantile(0.95, apartamentos_latencia_predicao_seconds) > 1` | 🟡 Warning |
| Friedman não significativo | `apartamentos_estatistica_friedman_significativo == 0` | ℹ️ Info |
| Outliers > 5% em feature | `apartamentos_dados_outliers_percentual > 5` | 🟡 Warning |
| Memória > 2GB | `apartamentos_sistema_memoria_uso_mb > 2048` | 🟡 Warning |

---

## 7. Arquivos de Referência

| Arquivo | Responsabilidade |
| :--- | :--- |
| [`coletor_prometheus.py`](../app_build/observabilidade_metricas/coletor_prometheus.py) | Declaração e implementação de todas as métricas |
| [`contrato_telemetria.py`](../app_build/observabilidade_metricas/contrato_telemetria.py) | Interface/contrato do sistema de telemetria |
| [`servico_telemetria.py`](../app_build/observabilidade_metricas/servico_telemetria.py) | Inicia o servidor HTTP de métricas (:8000) |
| [`detector_drift.py`](../app_build/observabilidade_metricas/detector_drift.py) | Cálculo de PSI, KS e Wasserstein |
| [`fabrica_etapas.py`](../app_build/orquestracao_pipeline/fabrica_etapas.py) | Ponto de coleta das métricas em cada etapa |
| [`observador_mlflow.py`](../app_build/rastreamento_mlflow/observador_mlflow.py) | Publica métricas de estatística, holdout e negócio via eventos |
| [`executor_esteira.py`](../app_build/orquestracao_pipeline/executor_esteira.py) | Publica métricas de pipeline por etapa |
| [`config_ob/prometheus.yml`](../config_ob/prometheus.yml) | Configuração de scrape do Prometheus |
| [`config_ob/dashboards/`](../config_ob/dashboards/) | JSONs dos dashboards Grafana |
| [`docker-compose.yaml`](../docker-compose.yaml) | Infraestrutura completa da stack de observabilidade |

## Observabilidade real do MLflow Serving

O dashboard **Previsão Imobiliária — Ribeirão Preto/SP**, UID `painel-previsao-imoveis`, inclui operação, entradas, previsões e identificação do modelo. Os painéis existentes de treino e logs foram preservados. O arquivo provisionado é `config_ob/dashboards/dashboard_geral_imobiliario.json`.

### Coleta e interpretação

O serving mantém os endpoints nativos do MLflow (`/invocations`, `/health`, `/ping`, `/version`) e acrescenta `/metrics`. O Prometheus coleta `mlflow-serving:8080` no job `mlflow_serving` a cada 5 segundos. Apenas `/invocations` incrementa os contadores de requisições e latência. O módulo de inicialização é `app_build.observabilidade_metricas.servidor_inferencia`, que utiliza a aplicação oficial do MLflow.

| Métrica | Significado |
| --- | --- |
| `apartamentos_serving_requisicoes_total{status}` | Chamadas HTTP por classe 2xx/3xx/4xx/5xx |
| `apartamentos_serving_latencia_segundos` | Histograma de duração HTTP; p50/p95/p99 |
| `apartamentos_serving_lote_imoveis` | Histograma do tamanho dos lotes JSON reconhecidos |
| `apartamentos_serving_entradas_total` | Imóveis recebidos em JSON, inclusive em chamadas rejeitadas |
| `apartamentos_serving_entrada_problemas_total{campo,motivo}` | Ausentes, localidades desconhecidas e metragem inválida |
| `apartamentos_serving_valor{zona,bairro}` | Histograma de preços previstos por imóvel (R$) |
| `apartamentos_serving_valor_m2{zona,bairro}` | Histograma de preços previstos por m² (R$/m²) |
| `apartamentos_serving_referencia_total{nivel}` | Uso de referência de bairro, zona ou global por disponibilidade |
| `apartamentos_serving_modelo_info` | Nome, versão fixa carregada, run e alias usado na inicialização |
| `apartamentos_serving_treinado_timestamp` | Fim do run associado, ou início se ainda estiver aberto |
| `apartamentos_serving_registrado_timestamp` | Data de criação da versão no Registry; não é a data de mudança do alias |
| `apartamentos_serving_carregado_timestamp` | Data de carga do modelo neste processo |
| `apartamentos_serving_telemetria_falhas_total` | Falhas de interpretação da coleta, sem alteração da resposta |

Histogramas expõem os sufixos `_bucket`, `_sum` e `_count`. A distribuição de preços usa buckets até R$ 5 milhões e a de preço por m² até R$ 25 mil, além de `+Inf`. Medianas e percentis são aproximações; valores acima do último limite finito reduzem a precisão dos quantis.

As médias por zona/bairro usam **soma dos preços / número de imóveis**, com `increase` no período selecionado. Assim, lotes de tamanhos diferentes têm o peso correto. O volume por grupo aparece ao lado das médias. `increase` extrapola as amostras de scrape e pode produzir contagens fracionárias; não substitui um registro contábil de transações. Antes de dois scrapes ou sem tráfego, taxas, médias e quantis podem ficar sem dados. Contadores são reiniciados junto com o processo.

Entradas são inspecionadas nos formatos `dataframe_records` e `dataframe_split`; outros formatos aceitos pelo MLflow mantêm o comportamento original, mas não alimentam os diagnósticos de campos/tamanho dos lotes. A contagem HTTP continua funcionando. Não são armazenados corpos de requisições nos logs ou nas métricas.

### Fontes que ainda não estão disponíveis

- **MAE, RMSE e viés em produção:** dependem de integrar o preço real da venda associado a cada previsão. As métricas `apartamentos_holdout_*` são avaliação offline e estão identificadas assim.
- **PSI de produção:** exige baseline versionado e janela de dados reais. O script `scripts/servico_monitor_drift.py` usa dados simulados e valores fixos; não representa monitoramento real do serving. Os painéis legados de drift estão marcados como diagnóstico da fonte `ml_service`.
- **Fallback por suficiência:** o enriquecimento vetorizado atual aplica fallback quando a localidade está ausente das referências. O painel de cobertura mede esse comportamento real, não uma decisão por tamanho mínimo de amostra.
- **Treino/holdout:** dependem do exportador na porta 8000 (`ml_service`). Com esse processo parado, o indicador de disponibilidade fica em zero e os demais painéis podem ficar sem dados.

### Aplicação das configurações

```bash
docker compose --profile servico_ml --profile serving up -d --no-deps mlflow-serving
docker kill --signal=HUP prometheus
```

O Grafana lê o JSON provisionado automaticamente a cada 10 segundos. A versão do modelo é fixada na inicialização, evitando que a identificação apresentada diverja da versão carregada caso o alias seja alterado depois. Reinicie o serving para carregar uma nova versão. Mantenha um único worker neste modo de coleta em memória.
