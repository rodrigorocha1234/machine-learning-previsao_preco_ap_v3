# Guia de Implantação e Operação em Produção (Deployment)
## Projeto: Previsão de Preços de Apartamentos - Ribeirão Preto / SP

---

## 1. Topologia da Arquitetura e Serviços Provisionados

A solução foi projetada sob uma infraestrutura baseada em microsserviços conteinerizados via **Docker Compose**, garantindo isolamento, reprodutibilidade, observabilidade e alta disponibilidade:

| Serviço | Contêiner | Porta Externa | Finalidade / Papel Arquitetural | Healthcheck Endpoint |
|---|---|---|---|---|
| **MLflow Model Serving** | `mlflow-serving` | `8080` | Serving HTTP de inferência em tempo real com o modelo Champion (`@champion`) | `http://localhost:8080/health` |
| **MLflow Tracking Server** | `mlflow-server` | `5000` | Registro de experimentos, runs, parâmetros, métricas e Model Registry | `http://localhost:5000/health` |
| **PostgreSQL** | `mlflow-postgres` | `5432` | Backend store relacional do MLflow | `pg_isready -U postgres` |
| **RustFS / S3 Storage** | `storage` | `9000` / `9001` | Object storage compatível com S3 para persistência de artefatos de modelos | `http://localhost:9000/minio/health/live` |
| **Prometheus** | `prometheus` | `9090` | Coleta e armazenamento de métricas de telemetria e latência | `http://localhost:9090/-/healthy` |
| **Grafana** | `grafana` | `3000` | Dashboards analíticos e executivos para monitoramento do modelo e negócio | `http://localhost:3000/api/health` |
| **Grafana Loki** | `loki` | `3100` | Ingestão e centralização de logs estruturados dos contêineres | `http://localhost:3100/ready` |
| **Grafana Alloy** | `alloy` | `12345` | Coleta e encaminhamento de logs Docker para o Loki | `http://localhost:12345/-/healthy` |

---

## 2. Pré-requisitos de Ambiente

1. **Docker Engine**: Versão 24.0+ com suporte a Compose v2.
2. **Python**: Versão 3.12+ (ambiente local via venv para execução dos pipelines e testes).
3. **Rede Local**: Sub-rede Docker dedicada `172.20.0.0/16` (`mlflow-network`).
4. **Variáveis de Ambiente**: Arquivo `.env` configurado na raiz do projeto com credenciais S3, PostgreSQL e MLflow.

---

## 3. Comandos de Inicialização e Ciclo de Vida

### 3.1 Inicializar a Pilha Completa de Infraestrutura
```bash
# Iniciar todos os serviços necessários em segundo plano
docker compose --profile dashboard --profile serving up -d
```

### 3.2 Verificar a Saúde dos Contêineres
```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### 3.3 Parar ou Reiniciar Serviços
```bash
# Reiniciar apenas o serviço de inferência (após atualização do modelo campeão)
docker compose restart mlflow-serving

# Parar todos os serviços preservando os dados persistentes
docker compose down
```

---

## 4. Execução do Pipeline de Treinamento e Seleção

O pipeline automatizado de 20 etapas executa a ingestão, validação de contrato de dados, isolamento criptográfico do cofre de holdout, análise exploratória, validação cruzada aninhada com tuning (`n_jobs: 3`), testes estatísticos (Friedman, Nemenyi, Shapiro-Wilk), seleção do modelo campeão, avaliação do holdout por hierarquia geográfica (Global -> Zona -> Bairro), rastreamento no MLflow e registro no Model Registry:

```bash
# Ativar o ambiente virtual Python 3.12
source .venv/bin/activate

# Executar o fluxo principal apontando para o servidor de rastreamento local
PYTHONPATH=. MLFLOW_TRACKING_URI=http://localhost:5000 python -m app_build.orquestracao_pipeline.fluxo_principal
```

---

## 5. Guia de Inferência em Produção (MLflow Model Serving :8080)

O serviço de serving expõe a API padrão do MLflow em `http://localhost:8080/invocations`.

### 5.1 Especificação do Payload de Entrada (JSON Split Format)
```json
{
  "dataframe_split": {
    "columns": [
      "Area_Privativa_m2",
      "Quartos",
      "Suites",
      "Vagas",
      "Banheiros",
      "Zona",
      "Bairro"
    ],
    "data": [
      [120.0, 3, 1, 2, 3, "Zona Sul", "Jardim Botânico"],
      [65.0, 2, 1, 1, 2, "Zona Leste", "Ribeirânia"],
      [180.0, 4, 3, 3, 4, "Zona Sul", "Jardim Olhos D'Água"]
    ]
  }
}
```

### 5.2 Exemplo de Chamada via `curl`
```bash
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": [
        "Area_Privativa_m2",
        "Quartos",
        "Suites",
        "Vagas",
        "Banheiros",
        "Zona",
        "Bairro"
      ],
      "data": [
        [120.0, 3, 1, 2, 3, "Zona Sul", "Jardim Botânico"]
      ]
    }
  }'
```

### 5.3 Resposta Esperada (Previsão de Preço em Reais)
```json
{
  "predictions": [
    785420.50
  ]
}
```

---

## 6. Dashboards e Observabilidade

### 6.1 Acesso ao Grafana
- **URL**: `http://localhost:3000`
- **Usuário Padrão**: `admin`
- **Senha Padrão**: `Admin@2026`
- **Dashboards Provisionados**:
  - `Previsão de Preços de Apartamentos - Visão Geral do Modelo` (`config_ob/dashboards/dashboard_geral_imobiliario.json`):
    - Volume total de predições em tempo real.
    - Latência de inferência p50, p95 e p99.
    - Distribuição de previsões e ticket médio em Ribeirão Preto.
    - Métricas de erro (RMSE, MAE) por Zona e Bairro.

### 6.2 Acesso ao Prometheus
- **URL**: `http://localhost:9090`
- **Métricas Chave**:
  - `ml_predicoes_total`: Contador total de predições realizadas.
  - `ml_inferencia_duracao_segundos`: Histograma de tempo de resposta.
  - `ml_erro_predicoes_total`: Contador de requisições com falha.

### 6.3 Acesso ao MLflow UI
- **URL**: `http://localhost:5000`
- **Experimento Principal**: `previsao_preco_apartamentos_ribeirao_preto`
- **Model Registry**: Modelo `previsao_preco_apartamento_modelo` com a versão promovida sob o alias `@champion`.

---

## 7. Procedimentos de Manutenção e Auditoria

1. **Auditoria de Conformidade**:
   ```bash
   PYTHONPATH=. python -m pytest tests/
   mypy --ignore-missing-imports --explicit-package-bases app_build
   ruff check app_build
   ```
2. **Verificação de Logs em Tempo Real**:
   ```bash
   docker logs -f mlflow-serving
   docker logs -f mlflow-server
   ```
