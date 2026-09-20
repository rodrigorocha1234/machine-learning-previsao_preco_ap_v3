# Spec — Grid Search e Catálogo de Modelos

## Padrões
- Strategy para cada família de modelo.
- Factory Method para construir estimador + grid.

## Interface
```python
class RegressionStrategy(Protocol):
    @property
    def name(self) -> str: ...
    def build_pipeline(self) -> Pipeline: ...
    def param_grid(self) -> dict[str, list]: ...
```

## Grids mínimos
### Ridge
- `alpha`: [0.01, 0.1, 1, 10, 100]

### Lasso
- `alpha`: [0.0001, 0.001, 0.01, 0.1, 1]
- `max_iter`: [10000]

### ElasticNet
- `alpha`: [0.0001, 0.001, 0.01, 0.1, 1]
- `l1_ratio`: [0.1, 0.25, 0.5, 0.75, 0.9]

### DecisionTreeRegressor
- `max_depth`: [None, 3, 5, 8, 12]
- `min_samples_split`: [2, 5, 10, 20]
- `min_samples_leaf`: [1, 2, 5, 10]

### RandomForestRegressor
- `n_estimators`: [200, 500]
- `max_depth`: [None, 8, 15]
- `min_samples_leaf`: [1, 2, 5]
- `max_features`: ["sqrt", 0.7, 1.0]

### GradientBoostingRegressor
- `n_estimators`: [100, 300]
- `learning_rate`: [0.01, 0.05, 0.1]
- `max_depth`: [2, 3, 5]
- `subsample`: [0.7, 1.0]

### HistGradientBoostingRegressor
- `learning_rate`: [0.03, 0.05, 0.1]
- `max_iter`: [100, 300]
- `max_leaf_nodes`: [15, 31, 63]
- `l2_regularization`: [0, 0.1, 1.0]

### SVR
- `kernel`: ["rbf", "linear"]
- `C`: [0.1, 1, 10, 100]
- `epsilon`: [0.01, 0.1, 0.5]
- `gamma`: ["scale", "auto"]

### KNeighborsRegressor
- `n_neighbors`: [3, 5, 7, 11, 15]
- `weights`: ["uniform", "distance"]
- `p`: [1, 2]

## Scoring do GridSearchCV
Primário: `neg_mean_absolute_error`.
Secundários podem ser registrados, mas o refit deve ocorrer por MAE.

## CV interno
`KFold(n_splits=5, shuffle=True, random_state=<seed do fold externo>)`.

## Requisitos
- `n_jobs=-1` quando seguro;
- capturar `best_params_`, `best_score_`;
- salvar tabela completa `cv_results_` como artefato;
- logar duração do tuning;
- nunca usar os dados do fold externo de teste no Grid Search.
