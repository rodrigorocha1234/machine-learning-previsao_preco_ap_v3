# Implantação e operação local

Execute os comandos na raiz do repositório. Este guia descreve o Compose existente; não representa uma instalação endurecida para exposição pública ou um ambiente de alta disponibilidade.

## Pré-requisitos

- Docker com Compose v2 e acesso à rede para baixar imagens/pacotes.
- Python 3.12+ no host, por causa das APIs de tipagem usadas pelo código.
- Arquivo `dados/bairro_final_v3_engineered.xlsx` e permissão de escrita em `storage-data/` e `observabilidade_data/`.
- `.env` local configurado. Não copie credenciais para a documentação ou para commits.

Não há manifesto central de dependências com versões fixadas. As bibliotecas utilizadas incluem pandas, NumPy, scikit-learn, SciPy, statsmodels, MLflow, PyYAML, openpyxl, pyarrow, XGBoost, LightGBM, prometheus-client, requests, psutil e boto3. Testes utilizam pytest; verificações de estilo utilizam Ruff. Use o ambiente já preparado ou monte um ambiente com versões compatíveis; a lista não é um lock de reprodução.

```bash
python3.12 -m venv .venv
# Instale as dependências compatíveis antes de executar os módulos abaixo.
```

O Compose instala dependências em tempo de inicialização em alguns serviços. Isso pode exigir downloads grandes, inclusive dependências de GPU do XGBoost, mesmo sem uso de GPU. A imagem MLflow pode usar outra versão de Python; o comando atual aplica compatibilidade para `typing.override`. Não presuma que host e contêiner têm ambientes idênticos.

## Configuração

`configs/pipeline.yaml` controla divisão, CV, avaliação, seleção e amostra. `configs/modelos.yaml` controla modelos ativos, parâmetros e buscas. Algumas opções do YAML ainda não são aplicadas de ponta a ponta; veja a [especificação](Technical_Specification.md).

As variáveis do Compose incluem `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `PGPORT`, `MLFLOW_VERSION`, `MLFLOW_BACKEND_STORE_URI`, `MLFLOW_HOST`, `MLFLOW_PORT`, `MLFLOW_S3_ENDPOINT_URL`, `MLFLOW_ARTIFACTS_DESTINATION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` e `S3_BUCKET`. Grafana também aceita `GRAFANA_ADMIN_USER` e `GRAFANA_ADMIN_PASSWORD`. Consulte o próprio YAML para defaults e uso exato.

No host, `ObservadorMlflow` usa `MLFLOW_TRACKING_URI` ou `http://localhost:5000`. A aplicação não carrega `.env` automaticamente como o Compose faz; configure o ambiente do processo/IDE quando necessário. Variáveis S3 podem ser necessárias conforme o modo de acesso aos artefatos.

## Iniciar serviços

```bash
# Infraestrutura de tracking, banco e storage
# Observabilidade inclui o exportador persistente metricas-treino
docker compose --profile servico_ml --profile dashboard config --quiet
docker compose --profile servico_ml --profile dashboard up -d
```

Na primeira instalação, o exportador responde 503 em `/metrics` até existir um snapshot. O serving precisa de um modelo registrado com alias `champion` antes de iniciar.

| Serviço | Perfil | Acesso |
| --- | --- | --- |
| PostgreSQL, RustFS, criação do bucket, MLflow | `servico_ml` | Portas do `.env`; storage 9000/9001; MLflow tipicamente 5000 |
| Grafana | `dashboard` | http://localhost:3000 |
| Prometheus | `dashboard` | http://localhost:9090 |
| Loki / Alloy | `dashboard` | 3100 / 12345 |
| Exportador de treino | `dashboard` | `metricas-treino:8000`, interno à rede Docker |
| MLflow Serving | `serving` | http://localhost:8080 |

## Treinar e publicar

```bash
.venv/bin/python -m app_build.orquestracao_pipeline.fluxo_principal
```

Executa as 20 etapas. A etapa 19 registra uma versão de `previsao_preco_apartamento_modelo` e atualiza `champion`. Confira no MLflow se o registro ocorreu: alguns erros de tracking são emitidos como avisos e não interrompem toda a execução.

```bash
# Inclua servico_ml para que as dependências do perfil serving sejam reconhecidas
docker compose --profile servico_ml --profile serving up -d --no-deps mlflow-serving
```

`servidor_inferencia` carrega a versão do alias na inicialização e usa a aplicação de scoring nativa MLflow com instrumentação. A etapa 20 do pipeline não sobe o contêiner.

Para carregar um alias atualizado sem alterar o Compose:

```bash
docker compose --profile servico_ml --profile serving restart mlflow-serving
```

Alterações no comando, ambiente ou volumes exigem `up -d`, não apenas `restart`. Reiniciar pode reinstalar/verificar dependências e interrompe brevemente a API. A telemetria em memória foi implementada para um worker.

## Recalcular avaliação sem promover outro modelo

```bash
.venv/bin/python -m scripts.recalcular_metricas
```

Executa novamente etapas 1–18: carga, staging, treino, Nested CV, avaliação de holdout e negócio. Registra resultados no MLflow e salva snapshot. Não executa a etapa de registro/promoção; o modelo servido permanece inalterado. Logo, as métricas dessa avaliação não devem ser automaticamente atribuídas à versão carregada na API.

Isso é uma nova avaliação, não uma recuperação de histórico. Não use repetidas inspeções do holdout para orientar ajustes de modelo. Cada execução cria seu próprio cofre; o controle de acesso único é por instância.

## Coleta e persistência

```bash
mkdir -p observabilidade_data
docker compose --profile dashboard up -d --no-deps metricas-treino
docker exec prometheus promtool check config /etc/prometheus/prometheus.yml
docker kill --signal=HUP prometheus
```

O snapshot `observabilidade_data/treino.prom` é salvo atomicamente no início e fim das etapas e na conclusão. O exportador o lê por volume somente leitura. Não apague o arquivo para “atualizar” os painéis: execute a avaliação se precisar de novos resultados. Execute um produtor por arquivo; não há resolução de concorrência entre treinamentos.

Grafana lê os dashboards de `config_ob/dashboards/` a cada 10 segundos. Após mudança de tipo/transformação de painel, recarregue a página. O dashboard geral está em [painel-previsao-imoveis](http://localhost:3000/d/painel-previsao-imoveis); o painel separado de localidades mostra holdout.

## Verificações

```bash
curl -f http://localhost:8080/health
curl -f http://localhost:8080/metrics
curl -f http://localhost:9090/-/healthy
curl -f http://localhost:3000/api/health
.venv/bin/python chamada.py
```

No Prometheus, confira `up{job="ml_service"}` e `up{job="mlflow_serving"}`. O `/health` do exportador testa o processo; `/metrics` testa a disponibilidade do snapshot. Para diagnosticar painéis vazios, veja [observabilidade](../docs/observabilidade.md).

## Estado e limitações operacionais

Preserve `pgdata/`, `storage-data/`, `prometheus_data/`, `grafana_data/` e `observabilidade_data/` conforme sua política de backup. Arquivos locais `mlruns/` e `mlflow.db` também podem existir, mas não demonstram que o servidor remoto usa esses mesmos armazenamentos.

O Compose inclui configurações permissivas, como acesso anônimo administrativo ao Grafana e hosts amplos no MLflow. Antes de qualquer exposição externa, revise autenticação, rede, TLS, segredos e backups. Dependências fixadas e imagens próprias permanecem pendentes. Estas são limitações da configuração presente, não uma certificação de segurança.
