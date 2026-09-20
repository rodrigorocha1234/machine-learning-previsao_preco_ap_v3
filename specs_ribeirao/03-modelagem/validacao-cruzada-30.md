# Spec — Validação Cruzada com 30 Avaliações

## Interpretação da regra “cross validate 30 vezes”
Usar:

```python
RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
```

Isso produz exatamente **30 folds externos de teste**.

## Fluxo por modelo e por fold externo
1. Separar treino e teste externo.
2. Criar GridSearchCV usando apenas treino externo.
3. Ajustar GridSearchCV com CV interno de 5 folds.
4. Recuperar `best_estimator_`.
5. Predizer no teste externo.
6. Calcular métricas.
7. Emitir eventos para os Observers.
8. Guardar resultados no formato longo.

## Métricas técnicas mínimas por fold
- MAE
- RMSE
- R²
- MAPE com proteção para zero
- Median Absolute Error
- Max Error

## Estrutura de resultados
| fold_id | repeat | fold | model | mae | rmse | r2 | mape | medae | max_error | best_params |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|

## Regra para estatística
Para Friedman/Nemenyi, construir matriz 30 × N modelos onde cada linha representa o mesmo fold externo e cada coluna um modelo. Usar MAE em escala positiva.

## Refit final
Após seleção estatística do campeão, executar novo GridSearchCV com o dataset completo e registrar o melhor pipeline final no MLflow.
