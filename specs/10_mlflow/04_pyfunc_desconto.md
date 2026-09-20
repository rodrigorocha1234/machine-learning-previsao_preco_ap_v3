# PyFunc para Previsão + Desconto

O modelo registrado no MLflow deve encapsular:

- pipeline campeão;
- schema;
- regra configurável de desconto;
- metadados de versão.

Exemplo conceitual:

```python
import mlflow.pyfunc
import pandas as pd

class ModeloImobiliarioPyFunc(mlflow.pyfunc.PythonModel):

    def carregar_contexto(self, contexto):
        ...

    def predict(self, contexto, entrada_modelo: pd.DataFrame):
        dados = entrada_modelo.copy()
        desconto = dados.pop("Percentual_Desconto").astype(float)

        self._validar_desconto(desconto)

        valor_previsto = self.modelo.predict(dados)
        valor_com_desconto = valor_previsto * (1 - desconto / 100.0)

        return pd.DataFrame({
            "Valor_Previsto": valor_previsto,
            "Percentual_Desconto": desconto,
            "Valor_Com_Desconto": valor_com_desconto
        })
```

Essa classe fica sozinha em `modelo_imobiliario_pyfunc.py`.
