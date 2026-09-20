# Spec — Modelo PyFunc com Métricas de Negócio

## Problema
O estimator scikit-learn retorna apenas previsão numérica. A API precisa devolver previsão + 30 métricas.

## Solução
Criar Adapter baseado em `mlflow.pyfunc.PythonModel`.

## Responsabilidade do Adapter
1. receber dataframe da API;
2. separar `Preco_Informado` das features;
3. validar entrada;
4. executar pipeline campeão;
5. calcular comparáveis e referências;
6. construir 30 métricas via `BusinessMetricsBuilder`;
7. devolver dataframe com colunas serializáveis.

## Pseudocontrato
```python
class RealEstatePyFuncModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # carregar pipeline e artefatos auxiliares
        ...

    def predict(self, context, model_input, params=None):
        # retorna DataFrame com métricas
        ...
```

## Artefatos auxiliares
- estatísticas por Zona;
- distribuição empírica por Zona;
- resíduos out-of-fold;
- referência de comparáveis;
- dispersão dos modelos candidatos;
- limites OOD;
- metadados da versão.

## Requisito
`Preco_Informado` nunca deve entrar no vetor de features do modelo. Ele existe somente para avaliação comercial do preço informado contra referências.
