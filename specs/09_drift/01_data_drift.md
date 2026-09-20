# Data Drift

## Baseline

Dataset usado pelo modelo campeão, versionado em `dados/referencia_drift/`.

## Numéricas

Métricas sugeridas:

- Kolmogorov-Smirnov;
- PSI;
- Jensen-Shannon após discretização;
- diferença de média/mediana;
- mudança de percentis;
- taxa de ausentes.

## Categóricas

- qui-quadrado;
- Jensen-Shannon;
- PSI categórico;
- novas categorias;
- categorias desaparecidas;
- mudança de cardinalidade;
- taxa de ausentes.

## Prediction drift

Comparar distribuição de `Valor_Previsto` atual com a baseline.

## Concept/target drift

Quando `Valor_da_Venda` real chegar, acompanhar degradação de RMSE/MAE/R² ao longo do tempo.

## Gate

O detector não retreina automaticamente. Ele publica um evento e registra:

- feature;
- estatística;
- p-valor quando aplicável;
- magnitude;
- severidade;
- data;
- versão do modelo.

## MLflow

Registrar relatório em run de monitoramento.


## Enum de severidade

Classificar o resultado com `NivelDrift`:

- `NivelDrift.ESTAVEL`
- `NivelDrift.ATENCAO`
- `NivelDrift.FORTE`

Não retornar strings livres de severidade.
