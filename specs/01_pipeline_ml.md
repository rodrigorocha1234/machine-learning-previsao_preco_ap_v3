# Spec 01 — Pipeline de Machine Learning

## Fluxo obrigatório
carregar configurações → validar configurações → carregar dados → validar dados → staging → separar DEVELOPMENT/HOLDOUT → bloquear HOLDOUT → EDA → EDA histórica → drift → Nested CV → testes estatísticos → seleção → tuning final → treinamento final → congelar configuração → abrir HOLDOUT → avaliação única GLOBAL/ZONA/BAIRRO → regras de negócio → MLflow → Registry → serving.

## Nested CV
- externa: RepeatedKFold, configuração inicial 5x3;
- interna: KFold;
- mesmas divisões externas para todos os modelos;
- validação externa não participa do tuning.

## Métricas
RMSE, MAE, MSE, R², RMSE relativo e MAPE. Métrica principal configurável por YAML.

## Estatística
Shapiro quando aplicável, ANOVA/Tukey/MultiComparison como complementares, Friedman sobre resultados externos e Nemenyi apenas quando Friedman for significativo.

## Seleção
Política configurável por YAML. Friedman/Nemenyi não escolhem sozinhos o modelo.
