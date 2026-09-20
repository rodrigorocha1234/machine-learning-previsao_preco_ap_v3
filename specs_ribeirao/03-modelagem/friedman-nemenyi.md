# Spec — Friedman e Nemenyi

## Objetivo
Testar se os modelos apresentam diferenças sistemáticas de desempenho nos mesmos 30 blocos de validação.

## Entrada
Matriz `scores` com:
- linhas = 30 folds externos;
- colunas = modelos;
- célula = MAE do modelo naquele fold.

## Friedman
Hipóteses:
- H0: os modelos têm ranks de desempenho equivalentes.
- H1: pelo menos um modelo difere.

Executar `scipy.stats.friedmanchisquare` nas colunas pareadas.

### Decisão
- `p >= 0.05`: não afirmar diferença estatística; selecionar por critérios de desempate previamente definidos.
- `p < 0.05`: executar Nemenyi pós-hoc.

## Nemenyi
Usar `scikit_posthocs.posthoc_nemenyi_friedman` com os 30 folds como blocos.

## Artefatos
- `friedman_result.json`: estatística, p-value, alpha;
- `nemenyi_pvalues.csv`;
- `mean_ranks.csv`;
- heatmap dos p-values;
- gráfico de ranks médios.

## Regra de seleção
1. Calcular rank por fold com menor MAE = melhor rank.
2. Obter rank médio.
3. Identificar o melhor rank médio.
4. Verificar no Nemenyi quais modelos não diferem significativamente dele.
5. Dentro desse conjunto, aplicar desempate:
   - menor MAE médio;
   - menor desvio-padrão do MAE;
   - menor RMSE;
   - menor complexidade;
   - maior interpretabilidade.

## Observação metodológica
O teste deve usar resultados pareados dos mesmos folds. Não usar 30 execuções independentes com splits diferentes por modelo.
