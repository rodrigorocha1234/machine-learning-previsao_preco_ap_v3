# Spec — Pré-processamento

## Classe principal
`ApartmentPreprocessor`

## Objetivo
Transformar dados mistos sem vazamento de informação e manter rastreabilidade dos nomes de features.

## Pipeline recomendado
### `Zona`
- imputação pela moda, se necessário;
- `OneHotEncoder(handle_unknown="ignore")`.

### Numéricas
`Quartos`, `Banheiros`, `Vagas_Garagem`, `Metragem`:
- imputação pela mediana;
- `StandardScaler` somente para modelos sensíveis à escala.

## Dois pipelines por Strategy
1. `scaled_pipeline`: Linear, Ridge, Lasso, ElasticNet, SVR, KNN.
2. `tree_pipeline`: DecisionTree, RandomForest, GradientBoosting, HistGradientBoosting.

## Requisito crítico
O `ColumnTransformer` deve estar dentro do `Pipeline` passado ao Grid Search. Nunca transformar o dataset completo antes da validação cruzada.

## Feature names
Após o fit, o módulo deve conseguir obter nomes como:

```text
cat__Zona_Centro
cat__Zona_Sul
num__Quartos
num__Banheiros
num__Vagas_Garagem
num__Metragem
```

Esses nomes serão usados na equação expandida dos modelos lineares e no relatório de interpretabilidade.

## Outliers
- não excluir automaticamente;
- sinalizar por IQR e MAD no EDA;
- permitir configuração de winsorização somente como experimento explícito;
- toda transformação deve ser treinada exclusivamente nos folds de treino.
