# Spec — Contrato funcional da API de previsão

## Objetivo
Padronizar o contrato entre o MLflow Model Serving e consumidores internos como CRM, painel de corretores, backoffice e rotinas batch.

## Separação de responsabilidades
### Modelo de ML
Recebe apenas features explicativas e devolve preço previsto.

### Camada PyFunc / domínio
Calcula métricas comerciais derivadas, valida desconto, cria `id_predicao`, anexa metadados do modelo e formata a resposta.

### Monitoramento
Calcula drift por lote/janela. O serving apenas reporta o último status conhecido ou `NAO_AVALIADO`.

## Códigos de alerta de precificação
- `SEM_ALERTA`
- `PRECO_INFORMADO_ACIMA_DA_ESTIMATIVA`
- `PRECO_INFORMADO_ABAIXO_DA_ESTIMATIVA`
- `PRECO_INFORMADO_DENTRO_DA_FAIXA`
- `SEM_PRECO_INFORMADO`

Os limites percentuais usados para classificar os alertas pertencem às regras de negócio e são configuráveis.

## Códigos de drift
- `SEM_ALERTA`
- `ALERTA_FEATURE_DRIFT`
- `ALERTA_PREDICTION_DRIFT`
- `ALERTA_PERFORMANCE_DRIFT`
- `NAO_AVALIADO`

## Campos de metadados
- `id_predicao`: UUID gerado em cada inferência;
- `versao_modelo`: versão registrada no MLflow;
- `alias_modelo`: ex. `champion`;
- `data_modelo`: data de promoção/registro;
- `status_drift`: último status consolidado.

## Campos de incerteza
Preferência: calibração conformal ou regressão quantílica separada.

- `preco_p10`
- `preco_p50`
- `preco_p90`

Regra: se não houver componente calibrado, retornar `null` e registrar `intervalo_disponivel=false`.

## Compatibilidade futura
Campos novos devem ser adicionados sem remover os existentes. Mudanças incompatíveis exigem nova versão de contrato.
