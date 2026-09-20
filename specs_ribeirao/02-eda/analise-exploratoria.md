# Spec — Análise Exploratória de Dados (EDA)

## Objetivo
Gerar um relatório reproduzível antes do treinamento, com visão técnica e visão de negócio.

## Seções obrigatórias
1. Resumo executivo.
2. Dimensão do dataset.
3. Tipos de dados.
4. Nulos por coluna.
5. Duplicatas.
6. Estatísticas descritivas.
7. Distribuição de `Valor_da_Venda`.
8. Distribuição de `Metragem`.
9. Preço por m² (`Valor_da_Venda / Metragem`).
10. Preço e preço/m² por `Zona`.
11. Preço por `Quartos`.
12. Preço por `Banheiros`.
13. Preço por `Vagas_Garagem`.
14. Relação Metragem × Valor.
15. Correlações das variáveis numéricas.
16. Cardinalidade e frequência de `Zona`.
17. Identificação de categorias raras.
18. Outliers por IQR e MAD.
19. Assimetria do target.
20. Possível necessidade de transformação log do target.
21. Segmentos com baixa amostra.
22. Possíveis inconsistências de negócio.
23. Riscos de viés amostral.
24. Recomendações para modelagem.

## Gráficos mínimos
- histograma e boxplot de Valor_da_Venda;
- histograma e boxplot de Metragem;
- histograma de valor/m²;
- scatter Metragem × Valor_da_Venda;
- boxplot de valor/m² por Zona;
- barras de contagem por Zona;
- heatmap de correlação numérica;
- medianas de preço por Quartos/Banheiros/Vagas.

## Estatísticas adicionais
Para cada Zona:
- contagem;
- média e mediana do preço;
- média e mediana do preço/m²;
- desvio-padrão;
- P25/P75;
- metragem mediana.

## Artefatos
- `eda_report.md` ou `eda_report.html`;
- `eda_summary.json`;
- tabelas CSV auxiliares;
- gráficos em PNG.

## Critério de aceite
O relatório nunca deve inventar conclusões. Se o dataset ainda não tiver sido fornecido, gerar estrutura/template e marcar resultados numéricos como pendentes.
