# Guia de Execução do Projeto — Previsão de Preços de Imóveis

Este guia apresenta o passo a passo completo para configurar, inicializar a infraestrutura, treinar o pipeline de Machine Learning, subir a API de inferência e monitorar a aplicação no Grafana.

---

## 1. Visão Geral da Arquitetura

O sistema é composto por:
- **Pipeline de Treinamento (Python 3.12)**: Executa 20 etapas determinísticas que englobam carga de dados, validação de regras, Nested Cross-Validation (5 folds × 3 repetições), tuning de hiperparâmetros ([configs/modelos.yaml](file:///home/rodrigo/PycharmProjects/machine-learning-previsao_preco_ap_v3/configs/modelos.yaml)), comitê de modelos (`VotingRegressor`), teste em holdout estrito e registro no MLflow.
- **Armazenamento e Tracking**: PostgreSQL (metadados do MLflow) e RustFS/S3 (artefatos e modelos salvos).
- **Servidor de Inferência (API REST)**: Contêiner `mlflow-serving` na porta `8080`, enriquecido com regras imobiliárias geográficas e métricas Prometheus.
- **Stack de Observabilidade**: Prometheus, Grafana, Loki e Alloy para monitoramento de drift, métricas de treino e telemetria de requisições.

---

## 2. Pré-requisitos

Antes de iniciar, certifique-se de possuir instalado no ambiente:

1. **Docker e Docker Compose v2**:
   ```bash
   docker --version
   docker compose version
   ```
2. **Python 3.12+**:
   ```bash
   python3 --version
   ```
3. **Ambiente virtual configurado (`.venv`)**:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   # Instale as dependências caso ainda não estejam no ambiente
   pip install -r requirements.txt  # ou pacotes essenciais listados em Deployment.md
   ```
4. **Base de dados imobiliária**:
   - O arquivo `dados/bairro_final_v3_engineered.xlsx` deve estar presente na pasta `dados/`.
5. **Arquivo de variáveis de ambiente (`.env`)**:
   - Verifique se o arquivo `.env` na raiz do projeto está preenchido com as configurações de banco, S3 e portas locais.

---

## 3. Passo a Passo de Execução

### Passo 1: Inicializar a Infraestrutura Base (Docker)

Suba os serviços de banco de dados, storage S3 local, MLflow Server e observabilidade com o comando:

```bash
docker compose --profile servico_ml --profile dashboard up -d
```

#### Serviços e Portas Disponíveis:

| Serviço | Contêiner | Porta no Host | Descrição / Acesso |
| :--- | :--- | :---: | :--- |
| **MLflow Tracking** | `mlflow-server` | `5000` | [http://localhost:5000](http://localhost:5000) (Interface Web) |
| **RustFS (S3)** | `storage` | `9000` / `9001` | Armazenamento de artefatos do MLflow |
| **PostgreSQL** | `mlflow-postgres` | `5432` | Banco relacional para o backend store do MLflow |
| **Grafana** | `grafana` | `3000` | [http://localhost:3000](http://localhost:3000) (`admin` / `Admin@2026`) |
| **Prometheus** | `prometheus` | `9090` | [http://localhost:9090](http://localhost:9090) |
| **Loki** | `loki` | `3100` | Coletor centralizado de logs |
| **Métricas de Treino** | `metricas-treino` | Interna | Expõe `observabilidade_data/treino.prom` ao Prometheus |

> [!NOTE]
> Aguarde alguns segundos até que os contêineres atinjam o status `healthy`. Você pode verificar com `docker ps`.

---

### Passo 2: Executar o Pipeline de Treinamento e Registrar o Modelo

Com a infraestrutura ativa, execute o pipeline completo na raiz do projeto:

```bash
.venv/bin/python -m app_build.orquestracao_pipeline.fluxo_principal
```

#### O que esse comando realiza:
1. Executa as **20 etapas determinísticas** da esteira.
2. Separa 20% dos dados para **Holdout estrito** (com semente 42) e 80% para desenvolvimento.
3. Roda a **Nested Cross-Validation** (5 folds externos × 3 repetições) para estimativa não enviesada do erro.
4. Realiza o **Tuning de Hiperparâmetros** para os modelos candidatos ativos em [configs/modelos.yaml](file:///home/rodrigo/PycharmProjects/machine-learning-previsao_preco_ap_v3/configs/modelos.yaml).
5. Treina o comitê final (`VotingRegressor`) combinando os modelos campeões.
6. Avalia no conjunto de holdout e calcula métricas técnicas (RMSE, MAE, R², MAPE) e de negócio.
7. Registra a nova versão do modelo no **MLflow Model Registry** (`previsao_preco_apartamento_modelo`) e atribui a tag/alias `champion`.
8. Salva o snapshot atômico das métricas em `observabilidade_data/treino.prom` para consumo imediato pelo Grafana.

---

### Passo 3: Iniciar o Servidor de Inferência (API de Serving)

Após o pipeline registrar o modelo com o alias `champion`, inicie o contêiner de serving:

```bash
docker compose --profile servico_ml --profile serving up -d --no-deps mlflow-serving
```

Se o modelo foi retreinado e você deseja recarregar a nova versão `champion` na API já em execução:

```bash
docker compose --profile servico_ml --profile serving restart mlflow-serving
```

#### Validar a integridade da API:
```bash
curl -f http://localhost:8080/health
```
*(Deve responder HTTP 200 com status saudável)*

---

### Passo 4: Realizar Previsões (Consumo da API)

A API espera requisições POST no endpoint `/invocations` contendo os atributos do apartamento: `Bairro`, `Zona`, `Quartos`, `Banheiros`, `Vagas_Garagem` e `Metragem`.

#### Opção A: Executar via Script Python

Execute o script de teste pronto incluído na raiz:

```bash
.venv/bin/python chamada.py
```

#### Opção B: Executar via cURL (Terminal)

```bash
curl -X POST "http://localhost:8080/invocations" \
     -H "Content-Type: application/json" \
     -d '{
       "dataframe_records": [
         {
           "Bairro": "Jardim Botânico",
           "Zona": "Zona Sul",
           "Quartos": 3,
           "Banheiros": 2,
           "Vagas_Garagem": 2,
           "Metragem": 85.0
         }
       ]
     }'
```

#### Estrutura do Retorno da API:
A resposta contém **32 campos** enriquecidos pelo motor imobiliário, divididos em:
- **Base**: `valor_previsto` (preço predito pelo modelo em R$) e `valor_m2_previsto` (preço do m²).
- **Referência Global (Ribeirão Preto)**: `global_mediana_mercado`, `global_media_mercado`, `global_mediana_m2_mercado`.
- **Referência da Zona**: `zona_mediana_mercado`, `zona_media_mercado`, `zona_mediana_m2_mercado`, `zona_valor_previsto`, `diferenca_perc_zona`, `zona_faixa_segura_piso`, `zona_faixa_segura_teto`.
- **Referência do Bairro**: `bairro_mediana_mercado`, `bairro_media_mercado`, `bairro_mediana_m2_mercado`, `bairro_valor_previsto`, `diferenca_perc_bairro`, `bairro_desconto_5`, `bairro_desconto_10`, `bairro_faixa_segura_piso`, `bairro_faixa_segura_teto`.

---

### Passo 5: Acompanhar Dashboards e Observabilidade

Abra o navegador e acesse as ferramentas de observabilidade:

1. **Grafana** — [http://localhost:3000](http://localhost:3000)
   - **Login**: `admin` / `Admin@2026` (ou acesso anônimo com privilégios de Admin se habilitado).
   - **Dashboards provisionados**:
     - *Painel Geral de Previsão de Imóveis*: Acompanha métricas de validação cruzada, holdout, latência de inferência, taxa de requisições e distribuições.
     - *Painel de Localidades*: Análise comparativa e desempenho por Bairro e Zona.
2. **MLflow Tracking Server** — [http://localhost:5000](http://localhost:5000)
   - Permite inspecionar parâmetros de tuning testados, artefatos gerados, curvas e versões do modelo no catálogo.
3. **Prometheus Targets** — [http://localhost:9090/targets](http://localhost:9090/targets)
   - Confirma a raspagem dos endpoints `http://mlflow-serving:8080/metrics` e `http://metricas-treino:8000/metrics`.

---

## 4. Operações Complementares e Manutenção

### Recalcular Avaliação Sem Promover Novo Modelo
Se você alterou parâmetros de avaliação ou deseja revalidar as métricas de holdout sem alterar a versão ativa em produção:

```bash
.venv/bin/python -m scripts.recalcular_metricas
```
*(Executa as etapas 1 a 18 e atualiza o snapshot do Grafana sem sobrescrever o modelo champion)*.

### Executar Testes Automatizados (QA)
Para rodar a suíte completa de testes unitários e de integração:

```bash
.venv/bin/pytest tests/ -v
```

### Detecção de Data Drift e Monitoramento Contínuo
Para iniciar o serviço que inspeciona o desvio estatístico nas requisições:

```bash
.venv/bin/python -m scripts.servico_monitor_drift
```

---

## 5. Resolução de Problemas Comuns (Troubleshooting)

- **Painel do Grafana exibe "No Data":**
  1. Verifique se o pipeline já foi executado pelo menos uma vez para gerar o arquivo `observabilidade_data/treino.prom`.
  2. Verifique o status do contêiner: `docker compose ps metricas-treino`.
  3. No Prometheus ([http://localhost:9090/targets](http://localhost:9090/targets)), verifique se o target `metricas-treino` está com status `UP`.

- **O contêiner `mlflow-serving` falha ao iniciar (`unhealthy` ou sai com erro):**
  - O contêiner de serving necessita obrigatoriamente que exista um modelo registrado com a tag/alias `champion`. Caso seja a primeira execução, rode primeiro o pipeline de treino (`Passo 2`) antes de subir o contêiner de serving.

- **Encerrar todos os serviços:**
  ```bash
  docker compose --profile servico_ml --profile dashboard --profile serving down
  ```
  *(Se desejar remover também os dados persistidos nos volumes, adicione a flag `-v`)*.
