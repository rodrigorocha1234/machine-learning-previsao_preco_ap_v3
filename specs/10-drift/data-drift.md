# Spec — Data drift

## Baseline
Salvar estatísticas da amostra de desenvolvimento usada no modelo campeão.

## Numéricas
- PSI;
- Kolmogorov-Smirnov;
- mudança de média/mediana;
- mudança de desvio padrão;
- taxa fora da faixa histórica.

## Categóricas
- Jensen-Shannon;
- taxa de categorias novas;
- variação de frequência;
- categorias desaparecidas.

## Prediction drift
Comparar distribuição de `preco_estimado`.

## Target/performance drift
Quando `Valor_da_Venda` real chegar:
- comparar RMSE/MAE/R² atuais contra baseline;
- disparar gatilho de retreino.

## Gatilhos
Configuráveis. Exemplo:
- PSI > 0.20;
- KS p-value < 0.01 + efeito material;
- categoria nova > 5%;
- RMSE piora > 15%.

Evitar retreino automático por um único teste sem tamanho de efeito e volume mínimo.

## Integração com serving
Drift é uma propriedade de lote/janela, não de uma única observação. O serviço de previsão deve retornar o último status consolidado conhecido (`SEM_ALERTA`, `ALERTA_*`) ou `NAO_AVALIADO`. Não classificar uma observação isolada como drift apenas por estar distante da média histórica.
