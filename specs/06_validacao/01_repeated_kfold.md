# Validação Cruzada com RepeatedKFold

## Configuração obrigatória

```python
RepeatedKFold(
    n_splits=5,
    n_repeats=30,
    random_state=42
)
```

`n_splits` pode ser configurável; `n_repeats=30` é requisito do projeto.

## Métricas técnicas por fold

- RMSE
- MAE
- MSE
- R²
- MAPE, se não houver alvo próximo de zero
- MedAE
- erro percentual absoluto por faixa de preço

## Armazenamento

Salvar uma linha por:

```text
modelo, repeticao, fold, rmse, mae, mse, r2, tempo_treino, tempo_predicao
```

Também salvar agregado por repetição:

```text
modelo, repeticao, rmse_medio, mae_medio, ...
```

A tabela agregada por repetição alimenta o Friedman.
