# Spec 02 — Modelos e Tuning

## Modelos
- Linear Regression
- Lasso
- Ridge
- Elastic Net
- Decision Tree Regressor
- Random Forest Regressor
- SVR
- MLPRegressor
- XGBoost Regressor
- LightGBM Regressor

## Estratégias de tuning
`grid`, `random` e `nenhum`, implementadas por Strategy. Arquitetura aberta a Bayesiano/Optuna/Halving sem alterar MainPipeline.

## Persistência MLflow
Cada modelo possui run pai. Folds externos e tuning são runs filhos. Todo tuning registra parâmetros, best params, scores e artifacts.

## Explicabilidade de configuração
Depois de cada treinamento, gerar interpretação dos hiperparâmetros efetivos em linguagem de negócio e salvar no MLflow.
