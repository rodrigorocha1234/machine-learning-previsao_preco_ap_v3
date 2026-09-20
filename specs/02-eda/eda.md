# Spec — EDA obrigatória antes do treino

Para cada coluna numérica:
- contagem, nulos, distintos;
- média, mediana, moda;
- desvio-padrão, variância;
- mínimo, máximo, amplitude;
- Q1, Q3, IQR;
- percentis 1, 5, 10, 25, 50, 75, 90, 95, 99;
- assimetria (skewness);
- curtose (excesso de Fisher);
- coeficiente de variação;
- MAD;
- quantidade de outliers por IQR;
- correlação Pearson e Spearman com o alvo quando aplicável.

Para categóricas:
- cardinalidade;
- moda;
- frequência absoluta e relativa;
- categorias raras;
- taxa de nulos;
- preço médio/mediano por categoria.

Gráficos:
- histogramas;
- boxplots;
- ECDF;
- scatterplots de numéricas vs alvo;
- matriz de correlação;
- preço por zona;
- distribuição do alvo;
- QQ-plot do alvo.

Artefatos:
- `eda_resumo.csv`
- `eda_categoricas.csv`
- gráficos `.png`
- `relatorio_eda.md`

Todos registrados no MLflow.
