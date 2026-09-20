# Spec — Friedman e Nemenyi

## Momento correto
Nunca usar no GridSearch.

Aplicar **somente após** obter scores externos pareados de todos os modelos na mesma sequência de folds do `RepeatedKFold`.

## Matriz
Linhas = folds externos.
Colunas = modelos.
Valor = métrica comparável; para erro, usar RMSE (menor é melhor).

## Friedman
H0: os modelos têm desempenho equivalente em ranks.

Se `p >= alpha`: não afirmar diferença estatística global.
Se `p < alpha`: executar Nemenyi.

## Nemenyi
Comparação post hoc par a par sobre os ranks.

Artefatos:
- `friedman.json`
- `nemenyi_pvalues.csv`
- `ranks_medios.csv`
- diagrama de diferença crítica opcional.

## Escolha do campeão
A decisão não deve depender apenas de significância:
1. desempenho externo;
2. estabilidade;
3. erro no holdout final;
4. latência;
5. interpretabilidade;
6. custo;
7. robustez a drift.

A regra de desempate deve estar em configuração e ser registrada.
