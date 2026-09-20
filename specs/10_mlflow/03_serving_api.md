# Serving via MLflow

## Regra

Não criar FastAPI na aplicação.

O modelo é servido por:

```bash
mlflow models serve \
  -m "models:/preco-imoveis@champion" \
  --host 0.0.0.0 \
  --port 5001
```

A URI e a porta devem ser configuráveis conforme o `docker-compose.yml` da raiz.

## Endpoint padrão

```text
POST /invocations
```

## Payload

O wrapper PyFunc recebe as features + `Percentual_Desconto`.

Exemplo:

```json
{
  "dataframe_records": [
    {
      "Zona": "Centro",
      "Quartos": 3,
      "Banheiros": 2,
      "Vagas": 2,
      "Metragem": 120.0,
      "Percentual_Desconto": 5.0
    }
  ]
}
```

## Resposta esperada

```json
{
  "predictions": [
    {
      "Valor_Previsto": 650000.0,
      "Percentual_Desconto": 5.0,
      "Valor_Com_Desconto": 617500.0
    }
  ]
}
```

## Validação

O wrapper deve rejeitar ou marcar como inválido:

- desconto abaixo do mínimo;
- desconto acima do máximo configurado;
- campos obrigatórios ausentes;
- tipos incompatíveis.

As regras de limite vêm da configuração de negócio, não dos hiperparâmetros do modelo.
