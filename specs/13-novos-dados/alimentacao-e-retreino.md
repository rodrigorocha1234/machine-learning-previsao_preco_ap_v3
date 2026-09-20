# Spec — Novos dados

## Novos imóveis sem preço realizado
Usar apenas para inferência e monitorar drift de features/predições.

## Novos imóveis com venda realizada
Adicionar ao repositório histórico com:
- `id_imovel`;
- `data_predicao`;
- `versao_modelo`;
- `preco_predito`;
- `valor_da_venda_real`;
- features originais.

## Política de retreino
Pode ser:
- por volume: +N vendas novas;
- temporal: mensal/trimestral;
- por performance drift;
- por data drift;
- híbrida.

## Segurança contra contaminação
- imutabilidade do dataset bruto;
- versionar snapshot;
- deduplicar por chave;
- validar schema;
- checar valores impossíveis;
- separar lote de monitoramento do lote de treino;
- registrar `dataset_hash` no MLflow.

## Futuro processamento distribuído
`AdaptadorBackendExecucao` permite substituir `BackendLocal` por:
- Spark;
- Ray;
- Dask;
sem mudar regras de negócio ou contratos de treino.

## Reconciliação das previsões
Persistir `id_predicao` e `id_imovel` na inferência. Quando `Valor_da_Venda` real chegar, reconciliar com a previsão correspondente e calcular erro realizado por versão do modelo, zona e janela temporal.
