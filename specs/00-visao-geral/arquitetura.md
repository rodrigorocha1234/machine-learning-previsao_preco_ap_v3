# Arquitetura

## Objetivo
Prever `Valor_da_Venda` de imóveis residenciais com rastreabilidade, comparação estatística de algoritmos, capacidade de explicação para negócio e serving nativo pelo MLflow.

## Princípios
- Uma classe por módulo Python.
- Toda transformação treinável fica dentro do `Pipeline`.
- Holdout final não participa de GridSearch, CV, Friedman ou Nemenyi.
- GridSearch não usa Friedman/Nemenyi.
- Friedman/Nemenyi usam apenas scores externos da validação cruzada.
- Todas as execuções relevantes são rastreadas no MLflow.
- O MLflow é integrado pelo padrão **Observer**.
- Regras de negócio não ficam dentro do algoritmo de ML.
- Backend de execução é abstraído para futura distribuição.

## Camadas
- `dominio`: regras e contratos de negócio.
- `dados`: carga, schema, preparação e EDA.
- `modelos`: fábricas e estratégias de regressão.
- `validacao`: GridSearch, CV, testes estatísticos e curva de aprendizado.
- `monitoramento`: drift.
- `interpretabilidade`: equações, coeficientes e importâncias.
- `mlops`: MLflow, registro e serving.
- `infraestrutura`: adaptadores de execução.
- `aplicacao`: fachada/orquestração.
