# Estratégia de Testes

## Unitários

- carregadores;
- imputação;
- encoding;
- escaladores;
- fábricas;
- cálculo de métricas;
- regras de desconto;
- pesos de votação;
- detectores de drift;
- Observer MLflow com mocks.

## Integração

- pipeline completa com dataset sintético;
- GridSearchCV;
- MLflow Tracking;
- Model Registry;
- PyFunc;
- endpoint `/invocations`.

## Testes estatísticos

- matriz correta por repetição;
- Friedman somente após CV;
- Nemenyi condicionado ao p-valor do Friedman;
- mesma partição para todos os modelos.

## Contratos

- schema de entrada;
- schema de saída;
- categorias novas;
- desconto fora dos limites.
