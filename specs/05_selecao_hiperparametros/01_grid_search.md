# Seleção de Hiperparâmetros com Grid Search

## Regra

1. Grid Search usa apenas o conjunto de desenvolvimento.
2. A métrica primária padrão é `neg_root_mean_squared_error`.
3. Registrar métricas secundárias.
4. Escolher os melhores hiperparâmetros de cada modelo.
5. Congelar esses parâmetros.
6. Executar depois o `RepeatedKFold`.
7. **Não executar Friedman nem Nemenyi sobre os resultados do Grid Search.**

## Exemplo

```python
busca = GridSearchCV(
    estimator=pipeline,
    param_grid=grade,
    scoring={
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    },
    refit="rmse",
    cv=5,
    n_jobs=-1,
    return_train_score=True,
)
```

## MLflow

Registrar:

- grade completa;
- melhor combinação;
- `best_score_`;
- `cv_results_` como CSV/JSON;
- tempo médio de fit;
- ranking;
- modelo vencedor do Grid Search;
- child runs para combinações quando autolog estiver habilitado.

Para registrar todas as combinações no MLflow, configurar o autolog de tuning sem limitar apenas aos melhores runs, quando o volume for aceitável.
