# Métricas da Imobiliária

Calcular usando o modelo campeão no holdout e, posteriormente, em produção.

## 1. Percentual de imóveis dentro de ±5%, ±10% e ±15%

```text
Cobertura_5
Cobertura_10
Cobertura_15
```

## 2. Erro monetário médio

MAE em R$.

## 3. Erro por faixa de preço

Exemplo:

- até R$ 300 mil;
- R$ 300–600 mil;
- R$ 600 mil–1 mi;
- acima de R$ 1 mi.

## 4. Erro por Zona

Permite detectar regiões onde o modelo é mais fraco.

## 5. Viés de avaliação

\[
Vies = média(\hat y-y)
\]

Positivo: tende a superestimar.
Negativo: tende a subestimar.

## 6. Percentual de desconto seguro

Uma definição configurável:

```text
desconto_seguro = desconto que mantém o valor final
dentro de uma margem de risco definida pela imobiliária
e pelo erro validado do modelo.
```

Não codificar esse percentual como constante fixa.

## 7. Margem de negociação estimada

Diferença entre `Valor_Previsto` e `Valor_Com_Desconto`.

## 8. Risco de subprecificação

Percentual de casos em que o preço após desconto ficaria abaixo de um limite de segurança baseado no erro histórico/intervalo de previsão.

## 9. Risco de superprecificação

Percentual de casos em que a recomendação fica acima da faixa aceitável.

## 10. Cobertura de confiança

Percentual das vendas reais que caem dentro da faixa prevista.

## 11. Receita potencial perdida por subavaliação

Somatório de diferenças negativas relevantes, quando houver valor real observado.

## 12. Tempo de venda por faixa de erro

Métrica futura se existir `Dias_Para_Vender`.

## Artefatos registrados diretamente no MLflow

- `metricas_imobiliaria.json`
- `metricas_por_zona.csv`
- `metricas_por_faixa_preco.csv`
- `cobertura_tolerancia.csv`
