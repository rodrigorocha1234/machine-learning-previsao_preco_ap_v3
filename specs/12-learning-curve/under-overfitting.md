# Spec — Underfitting / Overfitting

Gerar curva de aprendizado do campeão usando `sklearn.model_selection.learning_curve`.

Plotar:
- tamanho do treino;
- RMSE de treino;
- RMSE de validação;
- faixa ±1 desvio padrão.

Heurística:
- ambos erros altos e próximos: possível underfitting;
- treino muito melhor que validação, gap persistente: possível overfitting;
- ambos convergem para erro baixo: comportamento desejável.

Registrar:
- `curva_aprendizado.png`
- `curva_aprendizado.csv`
no run do campeão no MLflow.
