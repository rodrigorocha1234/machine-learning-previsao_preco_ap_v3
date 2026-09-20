# Grades Iniciais de Hiperparâmetros

As grades são ponto de partida e devem ser ajustadas ao tamanho do dataset.

```python
GRADES = {
    "ridge": {
        "modelo__alpha": [0.01, 0.1, 1, 10, 100]
    },
    "lasso": {
        "modelo__alpha": [0.0001, 0.001, 0.01, 0.1, 1]
    },
    "elastic_net": {
        "modelo__alpha": [0.001, 0.01, 0.1, 1],
        "modelo__l1_ratio": [0.1, 0.25, 0.5, 0.75, 0.9]
    },
    "arvore": {
        "modelo__max_depth": [3, 5, 8, 12, None],
        "modelo__min_samples_split": [2, 5, 10, 20],
        "modelo__min_samples_leaf": [1, 2, 5, 10]
    },
    "random_forest": {
        "modelo__n_estimators": [200, 500],
        "modelo__max_depth": [8, 12, 20, None],
        "modelo__min_samples_leaf": [1, 2, 5],
        "modelo__max_features": ["sqrt", 0.7, 1.0]
    },

    "gradient_boosting": {
        "modelo__n_estimators": [100, 200, 500],
        "modelo__learning_rate": [0.01, 0.05, 0.1],
        "modelo__max_depth": [2, 3, 5],
        "modelo__min_samples_leaf": [1, 2, 5],
        "modelo__subsample": [0.7, 0.9, 1.0]
    },
    "catboost": {
        "modelo__iterations": [200, 500],
        "modelo__depth": [4, 6, 8],
        "modelo__learning_rate": [0.01, 0.05, 0.1],
        "modelo__l2_leaf_reg": [1, 3, 5, 10],
        "modelo__verbose": [False]
    },
    "svr": {
        "modelo__kernel": ["rbf", "linear"],
        "modelo__C": [1, 10, 100],
        "modelo__epsilon": [0.01, 0.1, 0.5],
        "modelo__gamma": ["scale", "auto"]
    },
    "rede_neural": {
        "modelo__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
        "modelo__alpha": [0.0001, 0.001, 0.01],
        "modelo__learning_rate_init": [0.0005, 0.001, 0.01]
    },
    "xgboost": {
        "modelo__n_estimators": [200, 500],
        "modelo__max_depth": [3, 5, 8],
        "modelo__learning_rate": [0.01, 0.05, 0.1],
        "modelo__subsample": [0.7, 0.9, 1.0],
        "modelo__colsample_bytree": [0.7, 0.9, 1.0]
    },
    "lightgbm": {
        "modelo__n_estimators": [200, 500],
        "modelo__num_leaves": [15, 31, 63],
        "modelo__learning_rate": [0.01, 0.05, 0.1],
        "modelo__min_child_samples": [10, 20, 40],
        "modelo__subsample": [0.7, 0.9, 1.0]
    }
}
```
