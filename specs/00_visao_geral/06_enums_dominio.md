# Enums de Domínio

## Regra

Use `StrEnum` para categorias internas com conjunto fechado e conhecido de valores.

Evite strings mágicas em:

- seleção de modelos;
- factories;
- tipos de scaler;
- eventos do Observer;
- técnicas de ensemble;
- origem dos pesos;
- status de drift;
- tipos de carregador;
- métricas principais;
- estimadores finais do stacking.

Não use Enum para valores naturalmente abertos, como:

- `n_estimators`;
- `max_depth`;
- `alpha`;
- `learning_rate`;
- `Percentual_Desconto`;
- thresholds numéricos;
- caminhos;
- nomes de experimentos.

## Package

Criar:

```text
src/imobiliaria_ml/enums_pkg/
├── __init__.py
├── tipo_modelo.py
├── tipo_ensemble.py
├── tipo_escalonador.py
├── tipo_evento.py
├── nivel_drift.py
├── origem_peso_voting.py
├── tipo_carregador.py
├── tipo_metrica.py
└── tipo_estimador_final.py
```

Regra global continua válida:

```text
1 classe = 1 arquivo .py
```

## TipoModelo

```python
from enum import StrEnum

class TipoModelo(StrEnum):
    REGRESSAO_LINEAR = "regressao_linear"
    REGRESSAO_MULTIPLA = "regressao_multipla"
    REGRESSAO_POLINOMIAL = "regressao_polinomial"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    ARVORE_DECISAO = "arvore_decisao"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    SVR = "svr"
    REDE_NEURAL = "rede_neural"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
```

## TipoEnsemble

```python
class TipoEnsemble(StrEnum):
    VOTING_MEDIA = "voting_media"
    VOTING_PONDERADO_RMSE = "voting_ponderado_rmse"
    VOTING_PONDERADO_RANKING = "voting_ponderado_ranking"
    STACKING = "stacking"
    BAGGING = "bagging"
```

## TipoEscalonador

```python
class TipoEscalonador(StrEnum):
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUSTO = "robusto"
    SEM_ESCALA = "sem_escala"
```

## TipoEvento

```python
class TipoEvento(StrEnum):
    EDA_FINALIZADA = "eda_finalizada"
    GRIDSEARCH_INICIADO = "gridsearch_iniciado"
    GRIDSEARCH_FINALIZADO = "gridsearch_finalizado"
    VALIDACAO_FINALIZADA = "validacao_finalizada"
    FRIEDMAN_FINALIZADO = "friedman_finalizado"
    NEMENYI_FINALIZADO = "nemenyi_finalizado"
    ENSEMBLE_FINALIZADO = "ensemble_finalizado"
    MODELO_CAMPEAO = "modelo_campeao"
    METRICAS_NEGOCIO_FINALIZADAS = "metricas_negocio_finalizadas"
    DRIFT_DETECTADO = "drift_detectado"
    CURVA_APRENDIZADO_GERADA = "curva_aprendizado_gerada"
```

## NivelDrift

```python
class NivelDrift(StrEnum):
    ESTAVEL = "estavel"
    ATENCAO = "atencao"
    FORTE = "forte"
```

## OrigemPesoVoting

```python
class OrigemPesoVoting(StrEnum):
    RMSE = "rmse"
    RANKING = "ranking"
```

## TipoCarregador

```python
class TipoCarregador(StrEnum):
    CSV = "csv"
    EXCEL = "excel"
    BANCO = "banco"
```

## TipoMetrica

```python
class TipoMetrica(StrEnum):
    RMSE = "rmse"
    MAE = "mae"
    MSE = "mse"
    R2 = "r2"
    MEDAE = "medae"
    MAPE = "mape"
```

## TipoEstimadorFinal

```python
class TipoEstimadorFinal(StrEnum):
    REGRESSAO_LINEAR = "regressao_linear"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
```

## Uso nas APIs internas

Prefira:

```python
def criar_modelo(self, tipo_modelo: TipoModelo) -> EstrategiaModelo:
    ...
```

em vez de:

```python
def criar_modelo(self, nome_modelo: str) -> EstrategiaModelo:
    ...
```

E:

```python
def notificar(
    self,
    evento: TipoEvento,
    dados: EventoGridSearch,
) -> None:
    ...
```

em vez de strings livres.
