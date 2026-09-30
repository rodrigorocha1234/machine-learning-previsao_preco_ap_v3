# Rules 04 — Machine Learning

- Holdout final bloqueado até congelar configuração final.
- Tuning somente na CV interna durante Nested CV.
- Resultados da CV externa alimentam avaliação e comparação estatística.
- Preprocessamento dentro de Pipeline/ColumnTransformer.
- Parâmetros e espaços de tuning somente em YAML.
- Strategy para modelos, preprocessamentos e tuning.
- Observer para MLflow, métricas e logs.
- Tuning deve ser registrado no MLflow ao terminar.
- Cada treinamento deve produzir interpretação de parâmetros para negócio.
- Avaliação final única no holdout em GLOBAL/ZONA/BAIRRO.
