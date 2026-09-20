# Spec — Parâmetros e Interpretação para a Equipe de Negócios

## Objetivo
Gerar explicações que traduzam parâmetros técnicos em implicações práticas, sem confundir correlação preditiva com causalidade.

## Saída por modelo
```text
Modelo
Melhores hiperparâmetros
MAE médio ± desvio
RMSE médio
R² médio
Rank médio
Fórmula de predição
Parâmetros aprendidos
Interpretação de negócio
Cuidados/limitações
```

## Linear / Ridge / Lasso / ElasticNet
### Exigir
- intercepto;
- coeficiente de cada feature pós-transformação;
- equação expandida;
- sinal do coeficiente;
- magnitude na escala correta.

### Interpretação
Para atributos numéricos, explicar a variação estimada mantendo as demais features constantes. Se houver `StandardScaler`, converter o efeito para unidade original ou declarar que o coeficiente está por desvio-padrão.

Para `Zona` one-hot, cada coeficiente deve ser interpretado em relação à categoria de referência ou ao esquema de codificação efetivamente utilizado.

### Aviso obrigatório
Coeficiente preditivo não implica efeito causal do atributo no preço.

## Árvores e ensembles
### Exigir
- hiperparâmetros;
- feature importances quando disponíveis;
- permutation importance em conjunto de validação;
- partial dependence/ICE opcional para variáveis numéricas;
- regras de exemplo para árvore simples.

### Interpretação
Explicar quais características mais contribuíram para separar faixas de preço e onde há não linearidade.

## SVR
### Exigir
- kernel;
- C;
- epsilon;
- gamma;
- número de support vectors.

Para kernel linear, exportar coeficientes. Para RBF, declarar que não existe coeficiente simples por feature no espaço original e usar permutation importance/SHAP compatível se habilitado.

## KNN
### Exigir
- k;
- função de peso;
- métrica/distância;
- exemplo dos vizinhos de uma inferência.

### Interpretação
O preço resulta de imóveis semelhantes no espaço de atributos; sensível à escala e à densidade local.

## Linguagem para negócio
Evitar:
- “Metragem causa aumento de X reais”.

Preferir:
- “No modelo, mantendo as demais variáveis constantes, um aumento de 1 m² está associado a aproximadamente X reais na estimativa.”
