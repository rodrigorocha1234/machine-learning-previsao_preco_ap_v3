# Spec — Seleção e Refit do Modelo Campeão

## Entradas
- resultados dos 30 folds;
- ranks médios;
- Friedman;
- Nemenyi, se aplicável;
- métricas agregadas;
- complexidade e interpretabilidade por Strategy.

## Regra determinística
1. Ordenar por rank médio de MAE.
2. Se Friedman não significativo, considerar todos estatisticamente não diferenciados e aplicar desempate técnico.
3. Se Friedman significativo, manter o melhor rank e modelos cujo Nemenyi contra ele tenha `p >= 0.05`.
4. Desempatar por:
   1. menor MAE médio;
   2. menor desvio-padrão do MAE;
   3. menor RMSE médio;
   4. menor complexidade operacional;
   5. maior interpretabilidade.

## Refit
Executar GridSearchCV final no dataset completo apenas após a escolha da família campeã.

## Registro
Salvar:
- `champion_model_name`;
- `champion_params`;
- `selection_reason.json`;
- `training_dataset_fingerprint`;
- seed;
- versão das bibliotecas;
- timestamp.

## Alias
Ao registrar no MLflow Model Registry, atribuir alias operacional como `champion` conforme suporte do ambiente.
