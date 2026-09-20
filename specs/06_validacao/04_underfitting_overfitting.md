# Diagnóstico de Underfitting e Overfitting

Gerar curva de aprendizado usando `learning_curve`.

## Gráfico

Eixo X: quantidade de exemplos de treinamento.

Eixo Y: erro ou score.

Duas curvas:

- treino;
- validação.

## Interpretação

- treino ruim + validação ruim → possível underfitting;
- treino muito bom + validação muito pior → possível overfitting;
- curvas próximas e boas → melhor capacidade de generalização.

## Artefatos MLflow

- `curva_aprendizado.png`
- `curva_aprendizado.csv`
- `diagnostico_under_overfitting.json`

O gráfico e os dados da curva devem ser enviados em memória ao `ObservadorMLflow`, que os registra diretamente no MLflow. Não criar arquivo local persistente como etapa intermediária.
