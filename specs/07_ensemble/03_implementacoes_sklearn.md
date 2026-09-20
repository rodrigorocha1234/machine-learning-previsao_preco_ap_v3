# Implementações Concretas de Ensemble

As técnicas de ensemble devem utilizar implementações concretas disponíveis no ecossistema sklearn.

## 1. VotingRegressor

Usar:

```python
from sklearn.ensemble import VotingRegressor
```

A classe de domínio:

```text
votacao_regressor.py
    class VotacaoRegressor
```

deve encapsular `sklearn.ensemble.VotingRegressor`.

### Exemplo sem pesos

```python
voting = VotingRegressor(
    estimators=[
        ("random_forest", random_forest),
        ("xgboost", xgboost),
        ("gradient_boosting", gradient_boosting),
    ]
)
```

### Exemplo com pesos

```python
voting = VotingRegressor(
    estimators=[
        ("random_forest", random_forest),
        ("xgboost", xgboost),
        ("gradient_boosting", gradient_boosting),
    ],
    weights=pesos
)
```

Os pesos não devem ser hardcoded como regra padrão.

Fontes permitidas para pesos:

- RMSE validado;
- ranking médio pós-Friedman/Nemenyi.

## 2. StackingRegressor

Usar:

```python
from sklearn.ensemble import StackingRegressor
```

Classe:

```text
stacking_regressor_imobiliario.py
    class StackingRegressorImobiliario
```

Exemplo:

```python
from sklearn.linear_model import Ridge

stacking = StackingRegressor(
    estimators=[
        ("random_forest", random_forest),
        ("xgboost", xgboost),
        ("gradient_boosting", gradient_boosting),
    ],
    final_estimator=Ridge()
)
```

O `final_estimator` deve ser configurável.

Estimadores finais permitidos inicialmente:

- `LinearRegression`
- `Ridge`
- `Lasso`
- `ElasticNet`

O stacking deve usar previsões out-of-fold adequadas internamente.

O holdout final não pode ser usado para treinar o meta-modelo.

## 3. BaggingRegressor

Usar:

```python
from sklearn.ensemble import BaggingRegressor
```

Classe:

```text
bagging_regressor_imobiliario.py
    class BaggingRegressorImobiliario
```

Exemplo:

```python
bagging = BaggingRegressor(
    estimator=arvore_decisao,
    n_estimators=100,
    max_samples=0.8,
    bootstrap=True,
    random_state=42
)
```

Bagging deve ser tratado como ensemble homogêneo e não como votação entre modelos heterogêneos.

Estimadores-base iniciais permitidos:

- `DecisionTreeRegressor`
- `SVR` opcional
- outros apenas mediante configuração explícita.

## 4. GradientBoostingRegressor

Usar:

```python
from sklearn.ensemble import GradientBoostingRegressor
```

Classe:

```text
regressao_gradient_boosting.py
    class RegressaoGradientBoosting
```

Este modelo participa do catálogo normal de candidatos e pode participar de Voting/Stacking se ficar estatisticamente elegível.

## 5. RandomForestRegressor

Usar:

```python
from sklearn.ensemble import RandomForestRegressor
```

Pode participar de Voting/Stacking.

## 6. XGBRegressor

Usar:

```python
from xgboost import XGBRegressor
```

Pode participar de Voting/Stacking.

## 7. LGBMRegressor

Usar:

```python
from lightgbm import LGBMRegressor
```

Pode participar de Voting/Stacking.

## 8. CatBoostRegressor

Usar:

```python
from catboost import CatBoostRegressor
```

Classe:

```text
regressao_catboost.py
    class RegressaoCatBoost
```

Pode participar de Voting/Stacking.

## Regra estatística

Somente modelos estatisticamente elegíveis após Friedman/Nemenyi podem compor `VotingRegressor` ou `StackingRegressor`.

Bagging é avaliado como candidato próprio porque gera múltiplos estimadores derivados de um estimador-base.

## Regra de configuração

```yaml
ensemble:
  usar_votacao: true

  tecnicas_habilitadas:
    - voting_media
    - voting_ponderado_rmse
    - voting_ponderado_ranking
    - stacking
    - bagging

  stacking:
    estimador_final: ridge

  bagging:
    estimador_base: arvore_decisao
    n_estimators: 100
    max_samples: 0.8
    bootstrap: true
```


## Enums obrigatórios

Usar:

```python
from imobiliaria_ml.enums_pkg.tipo_ensemble import TipoEnsemble
from imobiliaria_ml.enums_pkg.origem_peso_voting import OrigemPesoVoting
from imobiliaria_ml.enums_pkg.tipo_estimador_final import TipoEstimadorFinal
```

A `FabricaEnsemble` deve receber `TipoEnsemble`.

A escolha dos pesos deve receber `OrigemPesoVoting`.

O Stacking deve receber `TipoEstimadorFinal`.
