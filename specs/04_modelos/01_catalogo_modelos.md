# Catálogo de Modelos

Modelos obrigatórios:

1. Regressão Linear
2. Regressão Linear Múltipla
3. Regressão Polinomial
4. Ridge
5. Lasso
6. Elastic Net
7. Árvore de Decisão
8. Random Forest
9. Gradient Boosting
10. SVR
11. Rede Neural
12. XGBoost
13. LightGBM
14. CatBoost

## Implementações concretas

| Modelo | Implementação |
|---|---|
| Linear | `sklearn.linear_model.LinearRegression` |
| Múltipla | `sklearn.linear_model.LinearRegression` |
| Polinomial | `sklearn.preprocessing.PolynomialFeatures` + `LinearRegression` |
| Ridge | `sklearn.linear_model.Ridge` |
| Lasso | `sklearn.linear_model.Lasso` |
| Elastic Net | `sklearn.linear_model.ElasticNet` |
| Árvore | `sklearn.tree.DecisionTreeRegressor` |
| Random Forest | `sklearn.ensemble.RandomForestRegressor` |
| Gradient Boosting | `sklearn.ensemble.GradientBoostingRegressor` |
| SVR | `sklearn.svm.SVR` |
| Rede Neural | `sklearn.neural_network.MLPRegressor` |
| XGBoost | `xgboost.XGBRegressor` |
| LightGBM | `lightgbm.LGBMRegressor` |
| CatBoost | `catboost.CatBoostRegressor` |

Todos implementam uma interface comum:

```python
class EstrategiaModelo(ABC):
    @abstractmethod
    def criar_modelo(self, parametros: dict):
        ...

    @abstractmethod
    def obter_grade_hiperparametros(self) -> dict:
        ...

    @abstractmethod
    def obter_nome(self) -> str:
        ...
```

A `FabricaModelos` escolhe a Strategy sem acoplar a pipeline ao algoritmo.

Regra: uma classe por arquivo `.py`.


## Enum de modelos

Usar `TipoModelo` em toda seleção interna.

Exemplo:

```python
def criar_modelo(
    self,
    tipo_modelo: TipoModelo
) -> EstrategiaModelo:
    ...
```

Não usar strings livres para identificar modelos.


## Strategy tipada com Generic

A interface deve preservar o tipo concreto do estimador:

```python
TModelo = TypeVar(
    "TModelo",
    bound=RegressorProtocol,
)

class EstrategiaModelo(Generic[TModelo]):
    @abstractmethod
    def criar_modelo(
        self,
        parametros: ParametrosModelo,
    ) -> TModelo:
        ...
```

Cada Strategy concreta deve especializar `TModelo`.
