# Spec — GridSearch + RepeatedKFold

## Desenho recomendado: Nested CV
### Outer CV
`RepeatedKFold(n_splits=10, n_repeats=30, random_state=42)`

Total: 300 avaliações externas por modelo.

### Inner CV
`KFold(n_splits=5, shuffle=True, random_state=...)`

Dentro de cada fold externo:
1. GridSearch só vê `X_treino_externo`.
2. Seleciona hiperparâmetros pela métrica principal.
3. Melhor pipeline prediz `X_teste_externo`.
4. Salvar métricas do fold externo.
5. Salvar `cv_results_` do GridSearch no MLflow.

## Métrica primária
`neg_root_mean_squared_error` para seleção.

## Métricas secundárias
- MAE
- RMSE
- R²
- MedAE
- MAPE (com cautela para alvo próximo de zero)
- sMAPE
- RMSLE quando alvo > 0
- erro percentual P50/P90/P95
- viés médio
- taxa dentro de ±5%, ±10%, ±15%, ±20%

## Holdout
Separar 15%–20% no começo. Usar apenas uma vez depois que campeão, pipeline e hiperparâmetros estiverem congelados.
