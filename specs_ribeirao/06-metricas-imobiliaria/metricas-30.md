# Spec — 30 Métricas para a Imobiliária

## Entrada necessária
Para que as 30 métricas sejam calculadas, a requisição deve conter as features do imóvel e `Preco_Informado`.

Definições:
- `P` = Preco_Informado
- `Y` = valor estimado pelo modelo
- `A` = Metragem
- `ppm2_P = P/A`
- `ppm2_Y = Y/A`

## Métricas
| # | Campo | Fórmula / definição | Interpretação de negócio |
|---:|---|---|---|
| 1 | valor_estimado | `Y` | referência de valor pelo modelo |
| 2 | preco_informado | `P` | preço de anúncio/proposta informado |
| 3 | gap_preco_estimativa | `P - Y` | diferença nominal contra o modelo |
| 4 | gap_preco_estimativa_pct | `(P-Y)/Y*100` | sobrepreço/desconto percentual vs modelo |
| 5 | indice_preco_estimativa | `P/Y` | 1,00 = alinhado ao modelo |
| 6 | valor_m2_informado | `P/A` | preço/m² anunciado |
| 7 | valor_m2_estimado | `Y/A` | preço/m² implícito da previsão |
| 8 | gap_valor_m2 | `ppm2_P-ppm2_Y` | diferença de preço/m² |
| 9 | gap_valor_m2_pct | `(ppm2_P-ppm2_Y)/ppm2_Y*100` | diferença percentual de preço/m² |
| 10 | limite_inferior_estimativa | calibração residual | piso da faixa de referência |
| 11 | limite_superior_estimativa | calibração residual | teto da faixa de referência |
| 12 | amplitude_intervalo | `upper-lower` | incerteza absoluta |
| 13 | amplitude_intervalo_pct | `(upper-lower)/Y*100` | incerteza relativa |
| 14 | percentil_preco_zona | ECDF de P na Zona | posição do preço na zona |
| 15 | percentil_valor_m2_zona | ECDF de `P/A` na Zona | posição do preço/m² na zona |
| 16 | mediana_preco_zona | mediana histórica | referência central da zona |
| 17 | gap_mediana_preco_zona_pct | `(P-mediana)/mediana*100` | desvio do preço frente à mediana da zona |
| 18 | media_preco_zona | média histórica | referência média da zona |
| 19 | mediana_valor_m2_zona | mediana histórica por m² | referência robusta por m² |
| 20 | gap_mediana_m2_zona_pct | `(ppm2_P-mediana_m2)/mediana_m2*100` | desvio de preço/m² frente à zona |
| 21 | media_valor_m2_zona | média histórica por m² | referência média por m² |
| 22 | qtd_imoveis_referencia_zona | contagem de treino na zona | força amostral da referência zonal |
| 23 | valor_por_quarto | `P/max(Quartos,1)` | valor nominal por dormitório |
| 24 | valor_por_banheiro | `P/max(Banheiros,1)` | valor nominal por banheiro |
| 25 | valor_por_vaga | `P/max(Vagas_Garagem,1)`; nulo se 0 | valor nominal por vaga |
| 26 | metragem_por_quarto | `A/max(Quartos,1)` | espaço disponível por dormitório |
| 27 | mediana_preco_comparaveis | mediana dos k imóveis mais similares | referência por imóveis comparáveis |
| 28 | gap_comparaveis_pct | `(P-mediana_comps)/mediana_comps*100` | diferença vs comparáveis |
| 29 | dispersao_modelos | desvio-padrão das previsões dos modelos candidatos refitados | grau de concordância entre algoritmos |
| 30 | score_confianca | score 0–100 baseado em cobertura, OOD, intervalo e dispersão | resumo operacional de confiança |

## Score de confiança
Não deve ser apresentado como probabilidade estatística. É um índice operacional documentado, por exemplo:

```text
100
- penalidade_intervalo
- penalidade_baixa_amostra_zona
- penalidade_out_of_distribution
- penalidade_discordancia_modelos
```

Cada penalidade deve ser calibrada e documentada.

## Comparáveis
Calcular sobre espaço padronizado de:
- Zona (match/codificação);
- Quartos;
- Banheiros;
- Vagas_Garagem;
- Metragem.

Usar apenas dados históricos de treino. `k` padrão: 10, configurável.

## Intervalo de referência
Preferir quantis de resíduos out-of-fold do modelo campeão ou método conformal. Nunca usar apenas `± RMSE` como se fosse intervalo probabilístico garantido.
