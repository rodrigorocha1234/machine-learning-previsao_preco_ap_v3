# EDA Obrigatória

A EDA ocorre **antes do treinamento**.

## Estatísticas numéricas

Para cada variável numérica:

- quantidade;
- ausentes;
- média;
- mediana;
- moda;
- mínimo;
- máximo;
- amplitude;
- variância;
- desvio-padrão;
- Q1;
- Q3;
- IQR;
- percentis 1, 5, 10, 25, 50, 75, 90, 95, 99;
- assimetria (skewness);
- curtose;
- coeficiente de variação;
- quantidade de zeros;
- quantidade de valores únicos.

## Variáveis categóricas

- cardinalidade;
- frequência absoluta;
- frequência relativa;
- moda;
- categorias raras;
- ausentes;
- categorias presentes em treino e ausentes no conjunto de referência;
- relação entre categoria e alvo.

## Relações

- matriz de correlação Pearson;
- Spearman;
- dispersões contra o alvo;
- boxplots por `Zona`;
- distribuição do alvo;
- `Valor_da_Venda / Metragem`;
- análise de multicolinearidade (VIF quando aplicável);
- outliers por IQR e, opcionalmente, Isolation Forest apenas para diagnóstico.

## Artefatos registrados diretamente no MLflow

- `estatisticas_descritivas.csv`
- `categorias.csv`
- `correlacao_pearson.png`
- `correlacao_spearman.png`
- `distribuicao_alvo.png`
- `boxplot_zona_valor.png`
- `relatorio_eda.json`


## Persistência

As tabelas e figuras devem permanecer em memória até o `ObservadorMLflow` registrá-las. Não salvar versões persistentes locais como etapa intermediária.
