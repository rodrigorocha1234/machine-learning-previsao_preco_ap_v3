# Spec — Contrato da Inferência MLflow

## Entrada
Uma ou mais linhas com:

```json
{
  "Zona": "Sul",
  "Quartos": 3,
  "Banheiros": 2,
  "Vagas_Garagem": 2,
  "Metragem": 95.0,
  "Preco_Informado": 650000.0
}
```

## Regras de validação
- `Zona` obrigatória e não vazia;
- `Quartos`, `Banheiros`, `Vagas_Garagem` >= 0;
- `Metragem` > 0;
- `Preco_Informado` > 0;
- zona desconhecida deve ser aceita pelo encoder, mas marcada como risco/baixa cobertura;
- valores fora do domínio de treino devem gerar flags, não erro automático, salvo limites físicos configurados.

## Saída
Cada linha deve retornar:
- identificador opcional da linha;
- nome/versão do modelo;
- 30 métricas;
- `posicionamento_preco` categórico;
- `flags` de qualidade/coverage.

## Posicionamento de preço
Faixas configuráveis usando gap e intervalo:
- `abaixo_da_faixa`
- `dentro_da_faixa`
- `acima_da_faixa`

Não usar linguagem prescritiva como “comprar” ou “vender”; o endpoint fornece referência analítica.

## Erros
Erros de domínio devem retornar mensagem serializável clara no contexto do MLflow PyFunc. Validar lote inteiro e apontar índice da linha problemática.
