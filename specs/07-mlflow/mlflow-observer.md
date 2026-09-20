# Spec — MLflow como Observer

## Eventos observáveis
- início/fim da EDA;
- início/fim do GridSearch;
- fim de cada fold externo;
- comparação estatística;
- seleção do campeão;
- avaliação holdout;
- geração de curva de aprendizado;
- interpretação;
- drift;
- registro/publicação.

## Observer
`ObservadorMLflow` implementa `ObservadorTreinamento`.

O orquestrador publica eventos sem conhecer detalhes do MLflow.

## O que registrar
Parâmetros:
- modelo;
- hiperparâmetros;
- seed;
- n_splits;
- n_repeats;
- scaler;
- encoder;
- versão do dataset.

Métricas:
- métricas de CV;
- holdout;
- latência;
- drift;
- métricas imobiliárias agregadas quando existirem.

Artefatos:
- `cv_results.csv`;
- curvas de aprendizado;
- Friedman/Nemenyi;
- coeficientes/importâncias;
- EDA;
- baseline de drift;
- schema;
- configuração.

Modelo:
- `mlflow.pyfunc` final com parâmetro de inferência `desconto_percentual`.
