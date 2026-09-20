# Exemplo — PreProcessador

```python
class PreProcessador:

    def __init__(self, tratador_categorico, estrategia_escalonamento):
        self.tratador_categorico = tratador_categorico
        self.estrategia_escalonamento = estrategia_escalonamento

    def construir_pipeline(self, colunas_numericas, colunas_categoricas):
        ...
```

A divisão de treino/holdout deve ficar em uma classe própria quando o projeto crescer. O `PreProcessador` não deve conhecer MLflow.
