# Métricas Técnicas

## RMSE

\[
RMSE=\sqrt{\frac{1}{n}\sum(y_i-\hat y_i)^2}
\]

Explicação: erro típico com penalidade maior para erros grandes.

## MAE

\[
MAE=\frac{1}{n}\sum|y_i-\hat y_i|
\]

Explicação: erro absoluto médio em reais.

## MSE

\[
MSE=\frac{1}{n}\sum(y_i-\hat y_i)^2
\]

Mais difícil de comunicar ao negócio por estar em reais ao quadrado.

## R²

\[
R^2=1-\frac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}
\]

Percentual relativo da variabilidade explicada pelo modelo; não deve ser interpretado isoladamente como precisão.

## Métrica primária

RMSE.

## Métrica de comunicação

MAE em reais + erro percentual por faixa de valor.


## Enum de métricas

Use `TipoMetrica` para seleção de métricas internas.

Exemplo:

```python
metrica_primaria = TipoMetrica.RMSE
```

A configuração YAML pode usar `"rmse"`, mas deve ser convertida para o enum ao entrar na camada de aplicação.
