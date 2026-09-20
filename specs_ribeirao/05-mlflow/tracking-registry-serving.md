# Spec — MLflow Tracking, Registry e Model Serving

## Regra
Não criar uma API FastAPI própria. A interface HTTP de produção deve ser a interface nativa do MLflow Model Serving.

## Registro
Modelo final deve ser empacotado como `mlflow.pyfunc.PythonModel` para conseguir devolver:
- previsão;
- métricas imobiliárias;
- classificação textual de posicionamento de preço;
- metadados do modelo.

## Artefatos do modelo
- pipeline campeão;
- `business_context.json`;
- `feature_schema.json`;
- `model_interpretation.json`;
- `residual_calibration.json`;
- tabela de comparáveis resumida ou índice necessário ao cálculo;
- relatório final.

## Signature
Entrada lógica por linha:
- `Zona: string`
- `Quartos: long`
- `Banheiros: long`
- `Vagas_Garagem: long`
- `Metragem: double`
- `Preco_Informado: double`

`Preco_Informado` não entra no estimador de preço; ele é usado apenas para calcular métricas de negócio.

## Serving
Comando conceitual:
```bash
mlflow models serve --model-uri "models:/apartamento-ribeirao@champion" --port 5000 --env-manager local
```

Endpoint de inferência:
```text
POST /invocations
```

## Exemplo de payload
```json
{
  "dataframe_records": [
    {
      "Zona": "Sul",
      "Quartos": 3,
      "Banheiros": 2,
      "Vagas_Garagem": 2,
      "Metragem": 95.0,
      "Preco_Informado": 650000.0
    }
  ]
}
```

## Exemplo conceitual de resposta
```json
{
  "predictions": [
    {
      "valor_estimado": 623500.0,
      "preco_informado": 650000.0,
      "gap_preco_estimativa": 26500.0,
      "gap_preco_estimativa_pct": 4.25,
      "...": "demais métricas"
    }
  ]
}
```

## Health
Usar endpoints disponibilizados pelo próprio servidor MLflow (`/ping` ou `/health`) para health check.
