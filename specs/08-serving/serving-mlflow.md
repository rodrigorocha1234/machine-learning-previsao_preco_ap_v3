# Spec — Serving somente com MLflow

Não usar FastAPI. O modelo campeão é empacotado como `mlflow.pyfunc.PythonModel` e servido com `mlflow models serve`.

## Endpoint
Usar o endpoint nativo do MLflow:

```text
POST /invocations
Content-Type: application/json
```

## Features preditivas mínimas
- `Zona`
- `Quartos`
- `Banheiros`
- `Vagas`
- `Metragem`

## Campos comerciais opcionais na linha
- `Preco_Informado`
- `Id_Imovel`

Esses campos **não entram no pipeline preditivo**. O PyFunc deve removê-los antes de executar `pipeline.predict()` e utilizá-los somente para métricas de negócio e rastreabilidade.

## Params de inferência
- `desconto_percentual`: `double`, default `0.0`;
- futuramente: `comissao_percentual`, `custos_fixos`, `margem_seguranca`.

O parâmetro deve constar na `ModelSignature` do MLflow.

## Resposta mínima
Para cada imóvel retornar:
- `id_predicao`;
- `id_imovel`, se enviado;
- `preco_estimado`;
- `preco_estimado_m2`;
- `desconto_percentual`;
- `desconto_valor`;
- `preco_com_desconto`;
- `comissao_estimada`;
- `versao_modelo`;
- `alias_modelo`;
- `data_modelo`;
- `status_drift`.

Quando `Preco_Informado` existir, acrescentar:
- `preco_informado`;
- `preco_informado_m2`;
- `diferenca_absoluta`;
- `diferenca_percentual`;
- `alerta_precificacao`.

Quando um estimador de incerteza estiver disponível, acrescentar:
- `preco_p10`;
- `preco_p50`;
- `preco_p90`.

Caso não exista intervalo calibrado, esses campos devem ser `null`; nunca fabricar um intervalo a partir do ponto estimado.

## Exemplo de requisição

```json
{
  "dataframe_records": [
    {
      "Id_Imovel": "RP-000123",
      "Zona": "Sul",
      "Quartos": 3,
      "Banheiros": 2,
      "Vagas": 2,
      "Metragem": 110.0,
      "Preco_Informado": 680000.0
    }
  ],
  "params": {
    "desconto_percentual": 5.0
  }
}
```

## Exemplo de resposta simulada

```json
{
  "predictions": [
    {
      "id_predicao": "4fbf0d90-670c-4ff8-a4cf-d28d2b443a76",
      "id_imovel": "RP-000123",
      "preco_estimado": 620000.0,
      "preco_estimado_m2": 5636.36,
      "preco_informado": 680000.0,
      "preco_informado_m2": 6181.82,
      "diferenca_absoluta": 60000.0,
      "diferenca_percentual": 9.68,
      "desconto_percentual": 5.0,
      "desconto_valor": 31000.0,
      "preco_com_desconto": 589000.0,
      "comissao_estimada": 29450.0,
      "alerta_precificacao": "PRECO_INFORMADO_ACIMA_DA_ESTIMATIVA",
      "preco_p10": 575000.0,
      "preco_p50": 620000.0,
      "preco_p90": 668000.0,
      "versao_modelo": "12",
      "alias_modelo": "champion",
      "data_modelo": "2026-09-20",
      "status_drift": "SEM_ALERTA"
    }
  ]
}
```

## Validação de desconto
Regra padrão:

```text
0 <= desconto_percentual <= desconto_maximo_configurado
```

Se exceder o limite, a inferência deve falhar de modo explícito, por exemplo:

```json
{
  "error": {
    "codigo": "DESCONTO_ACIMA_DO_LIMITE",
    "mensagem": "O desconto solicitado excede o limite configurado pela imobiliária."
  }
}
```

## Rastreabilidade
Cada previsão deve ser persistível com:
- `id_predicao`;
- `id_imovel`;
- timestamp;
- versão/alias do modelo;
- hash/schema de entrada;
- preço previsto;
- preço informado;
- desconto aplicado;
- valor real futuro quando disponível.

Isso permite medir performance depois que a venda for concluída.
