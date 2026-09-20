# Holdout Final

Mesmo usando RepeatedKFold, manter um holdout final opcional/recomendado para uma avaliação final não usada na escolha de hiperparâmetros.

Fluxo:

```text
dataset completo
├── desenvolvimento
│   ├── Grid Search
│   └── RepeatedKFold + Friedman/Nemenyi
└── holdout final
    └── avaliação única do campeão
```

O holdout não pode ser reutilizado para decidir hiperparâmetros, escolher scaler ou definir o ensemble.

Se o dataset for pequeno demais para um holdout confiável, documentar a exceção e utilizar validação aninhada como evolução.
