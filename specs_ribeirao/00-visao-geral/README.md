# Spec — Visão Geral do Módulo de Regressão Imobiliária

## Objetivo
Construir um módulo de Machine Learning orientado por especificações para estimar `Valor_da_Venda` de apartamentos em Ribeirão Preto/SP a partir de:

- `Zona`
- `Quartos`
- `Banheiros`
- `Vagas_Garagem`
- `Metragem`

O projeto deve produzir um modelo campeão estatisticamente sustentado, registrar experimentos no MLflow e disponibilizar inferência pela API nativa do MLflow Model Serving, sem aplicação FastAPI própria.

## Restrições obrigatórias
1. Não criar API com FastAPI, Flask ou framework web próprio.
2. Servir o modelo com `mlflow models serve` e `POST /invocations`.
3. Usar `GridSearchCV` para ajuste de hiperparâmetros.
4. Fazer avaliação externa em 30 partições: `RepeatedKFold(n_splits=10, n_repeats=3)`.
5. O tuning deve ocorrer dentro de cada fold externo para evitar vazamento de informação.
6. Comparar os modelos usando Friedman e, quando significativo, Nemenyi pós-hoc.
7. MLflow deve ser conectado ao treinamento por padrão GoF Observer.
8. O pipeline de pré-processamento deve ser ajustado somente com dados de treino de cada fold.
9. O artefato final deve expor previsão + 30 métricas imobiliárias.
10. Devem ser gerados relatórios de EDA e de comparação/interpretação dos modelos.

## Padrões GoF adotados
- **Strategy**: algoritmos de regressão intercambiáveis.
- **Factory Method**: construção de modelos e grids.
- **Template Method**: fluxo de treinamento padronizado.
- **Observer**: MLflow recebe eventos de treino/validação/seleção.
- **Facade**: ponto único para orquestrar o pipeline.
- **Adapter**: adaptação do modelo final para `mlflow.pyfunc.PythonModel`.
- **Builder**: composição da resposta de negócio com 30 métricas.

## Modelos mínimos
- LinearRegression
- Ridge
- Lasso
- ElasticNet
- DecisionTreeRegressor
- RandomForestRegressor
- GradientBoostingRegressor
- HistGradientBoostingRegressor
- SVR
- KNeighborsRegressor

Opcionalmente, implementações externas como XGBoost/LightGBM podem ser adicionadas como Strategies sem alterar o restante do fluxo.

## Critério principal de seleção
O ranking de comparação deve usar **MAE** como métrica principal, por ser diretamente interpretável em reais. RMSE, R², MAPE e MedAE são métricas secundárias.

A seleção final deve obedecer:
1. Friedman sobre MAE dos 30 folds externos.
2. Se `p < 0.05`, executar Nemenyi.
3. Identificar o conjunto de modelos sem diferença estatisticamente significativa em relação ao melhor rank médio.
4. Entre modelos estatisticamente equivalentes, desempatar por menor MAE médio, menor desvio-padrão do MAE, menor complexidade e maior interpretabilidade, nesta ordem.

## Saídas esperadas
- modelo campeão registrado no MLflow;
- assinatura de entrada/saída;
- artefatos com parâmetros, coeficientes/estrutura e interpretação;
- matriz de resultados dos 30 folds;
- Friedman + Nemenyi;
- relatório de EDA;
- relatório de modelos;
- previsão e 30 métricas de negócio no endpoint `/invocations`.
