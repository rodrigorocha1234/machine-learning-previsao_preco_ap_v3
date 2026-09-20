# Friedman + Nemenyi

## Ordem obrigatória

```text
Grid Search
→ parâmetros congelados
→ RepeatedKFold
→ agregação por repetição
→ Friedman
→ se p < alpha: Nemenyi
```

## Friedman

Hipótese nula:

> os modelos apresentam o mesmo desempenho em termos de ranking entre os blocos avaliados.

Usar os mesmos blocos/partições para todos os modelos.

```python
from scipy.stats import friedmanchisquare
```

`alpha` padrão: `0.05`.

## Nemenyi

Executar apenas se:

```python
p_friedman < alpha
```

Usar `scikit_posthocs.posthoc_nemenyi_friedman`.

## Métrica estatística

Padrão: RMSE agregado por repetição. Menor é melhor.

## Saídas

- `friedman.json`
- `nemenyi_pvalores.csv`
- `ranking_medio.csv`
- `grupo_estatisticamente_elegivel.csv`
- heatmap do Nemenyi

## Regra de seleção

1. identificar o melhor ranking médio;
2. identificar modelos que **não apresentam diferença significativa** para o melhor modelo no Nemenyi;
3. esses modelos formam o grupo elegível;
4. desempatar dentro do grupo por critérios técnicos e operacionais previamente definidos, sem reutilizar o holdout final para tuning.

Se Friedman não for significativo, todos permanecem estatisticamente não diferenciados; o desempate usa critérios técnicos/operacionais.
