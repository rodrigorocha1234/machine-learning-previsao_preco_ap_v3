# Guia de Execução do Pipeline de Treinamento

Este documento descreve detalhadamente como configurar, executar e monitorar o pipeline completo de treinamento do projeto de Machine Learning para previsão de preços de imóveis (`Valor_da_Venda`).

---

## 1. Visão Geral da Arquitetura

O pipeline de treinamento implementa o padrão **Template Method** em [`PipelineTreinamento`](file:///home/rodrigo/PycharmProjects/machine-learning-previsao_preco_ap_v3/src/imobiliaria_ml/pipeline_pkg/pipeline_treinamento.py), orquestrando as etapas de ponta a ponta:

```text
Entrada de Dados (CSV / Excel / Banco)
          │
          ▼
1. Validação de Esquema e Qualidade (regras estritas, nulos, duplicatas)
          │
          ▼
2. Análise Exploratória de Dados (EDA) e geração de gráficos
          │
          ▼
3. Separação de Holdout Final (80% treino / 20% teste)
          │
          ▼
4. Pré-processamento e Escalonamento (OneHotEncoder + Scaler sob medida)
          │
          ▼
5. Otimização de Hiperparâmetros (GridSearchCV individual por modelo)
          │
          ▼
6. Validação Cruzada Robusta (RepeatedKFold: 5 folds x 30 repetições)
          │
          ▼
7. Testes de Hipótese Estatística (Friedman + Nemenyi para Critical Distance)
          │
          ▼
8. Delimitação do Grupo Estatisticamente Elegível
          │
          ▼
9. Avaliação e Construção de Ensembles (Voting, Stacking, Bagging)
          │
          ▼
10. Seleção do Modelo Campeão e Treinamento Final no Dataset de Treino
          │
          ▼
11. Avaliação no Holdout e Cálculo das Métricas de Negócio Imobiliário
          │
          ▼
12. Registro do Modelo e Artefatos no MLflow (PyFunc com Regras de Desconto)
```

---

## 2. Pré-requisitos e Ambiente

### 2.1. Ambiente Python
Certifique-se de que o ambiente virtual está ativo e as dependências instaladas:

```bash
# Ativação do ambiente virtual
source .venv/bin/activate

# Validação das dependências
pytest tests/ -v
```

### 2.2. Infraestrutura de Suporte (Docker Compose)
O projeto utiliza uma stack com **PostgreSQL** (backend store), **MinIO** (S3 artifact store), **MLflow Tracking Server** e **MLflow Serving**.

Inicie os serviços com:

```bash
# Iniciar todos os serviços em segundo plano
docker compose up -d

# Verificar se os containers estão saudáveis (healthy)
docker compose ps
```

Portas expostas:
- **MLflow Tracking Server**: `http://localhost:5000`
- **MinIO Console**: `http://localhost:9001` (Usuário: `minio` / Senha: `minio123`)
- **MinIO API (S3)**: `http://localhost:9000`
- **PostgreSQL**: `localhost:5432` (Usuário: `mlflow` / Senha: `mlflow` / DB: `mlflow`)
- **MLflow Model Serving (Inferência)**: `http://localhost:5002`

---

## 3. Formas de Execução

### Forma 1: Linha de Comando (CLI) — Recomendada

Você pode disparar o treinamento diretamente via módulo Python utilizando o orquestrador [`Treinar`](file:///home/rodrigo/PycharmProjects/machine-learning-previsao_preco_ap_v3/src/imobiliaria_ml/aplicacao_pkg/treinar.py):

#### Execução Padrão (usa `dados/bruto/imoveis.csv` e `configuracao.yaml`):
```bash
python -m src.imobiliaria_ml.aplicacao_pkg.treinar
```

#### Execução Customizada (especificando arquivo de dados e configuração):
```bash
python -m src.imobiliaria_ml.aplicacao_pkg.treinar \
  --dados dados/bruto/imoveis.csv \
  --config configuracao.yaml
```

Para visualizar todas as opções disponíveis:
```bash
python -m src.imobiliaria_ml.aplicacao_pkg.treinar --help
```

---

### Forma 2: Execução Programática em Python

Para invocar o treinamento dentro de scripts, notebooks ou rotinas agendadas:

```python
from pathlib import Path
from src.imobiliaria_ml.aplicacao_pkg.treinar import Treinar

# 1. Instancia o orquestrador com o arquivo de configuração
treinador = Treinar(caminho_configuracao="configuracao.yaml")

# 2. Executa o treinamento completo
resultado = treinador.executar_treinamento(caminho_dados="dados/bruto/imoveis.csv")

# 3. Inspeciona o resultado do campeão
print(f"Modelo Campeão: {resultado.nome_modelo}")
print(f"Tipo: {resultado.tipo_modelo.value}")
print(f"RMSE no Holdout: R$ {resultado.metricas_holdout.rmse:,.2f}")
print(f"R² no Holdout: {resultado.metricas_holdout.r2:.4f}")
print(f"Run ID MLflow: {resultado.run_id}")
```

---

## 4. Parâmetros de Configuração (`configuracao.yaml`)

O comportamento do treinamento é 100% parametrizável através de [`configuracao.yaml`](file:///home/rodrigo/PycharmProjects/machine-learning-previsao_preco_ap_v3/configuracao.yaml):

```yaml
projeto:
  seed: 42                           # Semente para reprodutibilidade estrita
  alvo: Valor_da_Venda               # Nome da variável alvo contínua

validacao:
  n_splits: 5                        # Quantidade de folds na validação cruzada
  n_repeats: 30                      # Repetições no RepeatedKFold (30 conforme spec)
  alpha_friedman: 0.05               # Nível de significância para o teste de Friedman
  alpha_nemenyi: 0.05                # Nível de significância para o pós-teste de Nemenyi
  metrica_primaria: rmse             # Métrica de ordenação (rmse, mae, mape, r2)

ensemble:
  usar_votacao: false                # Se true, avalia Voting/Stacking com modelos elegíveis
  tecnicas_habilitadas:
    - voting_media                   # Votação simples pela média aritmética
    - voting_ponderado_rmse          # Votação com pesos inversos ao RMSE
    - voting_ponderado_ranking       # Votação com pesos baseados no ranking
    - stacking                       # StackingRegressor com estimador final
    - bagging                        # BaggingRegressor
  stacking:
    estimador_final: ridge           # Modelo meta-aprendedor do stacking
  bagging:
    estimador_base: arvore_decisao
    n_estimators: 100
    max_samples: 0.8
    bootstrap: true

mlflow:
  tracking_uri_env: MLFLOW_TRACKING_URI   # Variável de ambiente do servidor MLflow
  experimento: previsao-preco-imoveis     # Nome do experimento
  nome_modelo: preco-imoveis              # Nome no Model Registry
  persistencia_local_artefatos: false     # Envio em memória (sem arquivos físicos locais)

negocio:
  desconto_minimo: 0.0                    # Desconto mínimo aceito (%)
  desconto_maximo_automatico: 5.0         # Margem de desconto concedida automaticamente (%)
  desconto_maximo_com_aprovacao: 10.0     # Desconto teto dependente de aprovação (%)

drift:
  psi_atencao: 0.10                  # Limiar PSI para alerta moderado de drift
  psi_forte: 0.25                    # Limiar PSI para alerta crítico de drift
```

---

## 5. Dica: Modo Rápido (Smoke Test) vs. Modo Completo Oficial

O treinamento completo oficial avalia **14 algoritmos de regressão** ao longo de **150 folds (5 splits x 30 repetições)**, totalizando mais de 2.100 ajustes de modelos com GridSearch e validação. Isso garante robustez estatística de produção, mas pode levar vários minutos para rodar.

### Para testes rápidos de homologação / desenvolvimento:
Basta alterar temporariamente em `configuracao.yaml`:
```yaml
validacao:
  n_splits: 3
  n_repeats: 2
```
Ou instanciar o `PipelineTreinamento` definindo os modelos a serem avaliados:
```python
from src.imobiliaria_ml.pipeline_pkg.pipeline_treinamento import PipelineTreinamento
from src.imobiliaria_ml.enums_pkg.tipo_modelo import TipoModelo
from src.imobiliaria_ml.configuracao_pkg.leitor_configuracao import LeitorConfiguracao

config = LeitorConfiguracao().carregar_do_arquivo("configuracao.yaml")
pipeline = PipelineTreinamento(
    configuracao=config,
    modelos_selecionados=[
        TipoModelo.REGRESSAO_LINEAR,
        TipoModelo.RIDGE,
        TipoModelo.RANDOM_FOREST
    ]
)
resultado = pipeline.executar("dados/bruto/imoveis.csv")
```

---

## 6. Acompanhamento no MLflow UI

Após iniciar ou concluir o treinamento, acesse a interface web do MLflow:

👉 **[http://localhost:5000](http://localhost:5000)**

### O que você encontrará no MLflow:
1. **Experimento `previsao-preco-imoveis`**:
   - **Parâmetros**: Seed, hiperparâmetros otimizados no Grid Search, estimador final, etc.
   - **Métricas Técnicas**: `rmse`, `mae`, `mape`, `r2` (médias de validação cruzada e avaliação final no holdout).
   - **Métricas de Negócio**: `desconto_seguro_medio`, `perda_margem_estimada`, `impacto_financeiro_potencial`.
   - **Estatísticas de Friedman/Nemenyi**: Estatística Chi-quadrado, p-valor e Critical Distance ($CD$).
2. **Artefatos (Gerados 100% em memória)**:
   - `eda/correlacao_pearson.png` e `eda/correlacao_spearman.png`.
   - `eda/distribuicao_alvo.png` e `eda/boxplot_zona_valor.png`.
   - `explicabilidade/importancia_features.png` ou `explicabilidade/equacao_modelo.txt`.
   - `estatistica/matriz_metricas_cv.json`.
3. **Model Registry (`preco-imoveis`)**:
   - Versão registrada do modelo com a tag `champion`.
   - Encapsulado como **MLflow PyFunc** contendo a lógica de previsão e o cálculo das regras de negócio imobiliário.

---

## 7. Testando o Modelo em Produção (Serving & Inferência)

O container `mlflow-serving` (configurado em `docker-compose.yaml`) expõe a API padrão do MLflow na porta **5002**:

### 7.1. Chamada de Inferência via cURL
A inferência aceita os dados dos imóveis e o parâmetro de negócio `Percentual_Desconto`:

```bash
curl -X POST http://localhost:5002/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [
      {
        "Zona": "Zona Sul",
        "Quartos": 3,
        "Banheiros": 2,
        "Vagas": 2,
        "Metragem": 95.0,
        "Percentual_Desconto": 5.0
      },
      {
        "Zona": "Zona Norte",
        "Quartos": 2,
        "Banheiros": 1,
        "Vagas": 1,
        "Metragem": 60.0,
        "Percentual_Desconto": 3.5
      }
    ]
  }'
```

### 7.2. Contrato de Resposta Retornado
A API retorna os valores monetários calculados e o desconto validado pelas regras de negócio:

```json
[
  {
    "Valor_Previsto": 450000.0,
    "Percentual_Desconto": 5.0,
    "Valor_Com_Desconto": 427500.0
  },
  {
    "Valor_Previsto": 280000.0,
    "Percentual_Desconto": 3.5,
    "Valor_Com_Desconto": 270200.0
  }
]
```

---

## 8. Monitoramento de Data Drift

Após colocar o modelo em produção, novos dados podem sofrer deriva (Data Drift ou Concept Drift). O módulo [`AvaliarDrift`](file:///home/rodrigo/PycharmProjects/machine-learning-previsao_preco_ap_v3/src/imobiliaria_ml/aplicacao_pkg/avaliar_drift.py) pode ser disparado para comparar os dados de referência com a base de produção:

```bash
python -c '
from src.imobiliaria_ml.aplicacao_pkg.avaliar_drift import AvaliarDrift
monitor = AvaliarDrift()
resultado = monitor.avaliar("dados/bruto/imoveis.csv", "dados/bruto/imoveis.csv")
print("Status de Drift:", resultado.nivel.value)
'
```

---

## 9. Solução de Problemas (Troubleshooting)

| Sintoma | Causa Mais Provável | Ação Recomendada |
|---|---|---|
| `Connection refused: localhost:5000` | O container do MLflow está parado | Execute `docker compose up -d mlflow` e verifique os logs com `docker compose logs -f mlflow`. |
| `S3UploadFailedError` ou erro MinIO | Bucket não criado ou credenciais incorretas | Verifique se o container `mlflow-create-bucket` concluiu a criação do bucket `mlflow` via `docker compose logs mlflow-create-bucket`. |
| `ColumnNotFoundError` na validação | Dados de entrada faltando colunas obrigatórias | O dataset deve conter: `Zona`, `Quartos`, `Banheiros`, `Vagas`, `Metragem` e `Valor_da_Venda`. |
| Treinamento demorado | Avaliação de 14 modelos com 30 repetições | Para testes rápidos, reduza `n_repeats` no `configuracao.yaml` para 2 ou selecione uma lista reduzida de modelos. |
