# Exemplo — Avaliador de Regressão

O exemplo original usa `accuracy_score`, que é uma métrica de classificação. Para previsão de preços, usar métricas de regressão.

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import numpy as np

class CalculadorMetricas:

    def calcular(self, valores_reais, previsoes):
        mse = mean_squared_error(valores_reais, previsoes)
        return {
            "mae": mean_absolute_error(valores_reais, previsoes),
            "mse": mse,
            "rmse": np.sqrt(mse),
            "r2": r2_score(valores_reais, previsoes),
        }
```
