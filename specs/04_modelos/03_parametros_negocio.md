# Hiperparâmetros Traduzidos para o Negócio

## Linear / Múltipla

- `fit_intercept`: permite um valor-base quando todas as features numéricas estão no zero de referência.
- `positive`: restringe coeficientes a positivos; somente usar se fizer sentido econômico e estatístico.

## Polinomial

- `degree`: complexidade das curvas e interações. Grau alto pode memorizar ruído.
- `include_bias`: controla termo constante gerado pela transformação.

## Ridge

- `alpha`: força de regularização. Maior valor = modelo mais conservador em relação a coeficientes extremos.

## Lasso

- `alpha`: maior valor = maior pressão para eliminar efeitos fracos.
- `max_iter`: limite computacional para convergência.

## Elastic Net

- `alpha`: intensidade total da regularização.
- `l1_ratio`: equilíbrio Lasso/Ridge. Próximo de 1 favorece Lasso; próximo de 0 favorece Ridge.

## Árvore

- `max_depth`: número máximo de níveis de decisão.
- `min_samples_split`: quantidade mínima de imóveis para abrir nova regra.
- `min_samples_leaf`: quantidade mínima de imóveis sustentando uma conclusão.
- `max_features`: quantas características podem ser consideradas por divisão.

## Random Forest

- `n_estimators`: quantidade de árvores.
- `max_depth`: profundidade máxima por árvore.
- `min_samples_leaf`: imóveis mínimos por folha.
- `max_features`: diversidade permitida entre árvores.

## SVR

- `C`: penalidade por erros; alto `C` tende a tentar ajustar mais os dados.
- `epsilon`: faixa de erro tolerada sem penalização.
- `gamma`: alcance de influência de cada observação no kernel RBF.
- `kernel`: forma da relação não linear.

## Rede Neural

- `hidden_layer_sizes`: capacidade da rede.
- `activation`: transformação usada pelos neurônios.
- `alpha`: regularização L2.
- `learning_rate_init`: tamanho inicial do passo de aprendizado.
- `max_iter`: limite de épocas/iterações.

## XGBoost

- `n_estimators`: quantidade de árvores sequenciais.
- `max_depth`: complexidade de cada árvore.
- `learning_rate`: contribuição de cada nova árvore.
- `subsample`: proporção de imóveis usada por árvore.
- `colsample_bytree`: proporção de features por árvore.
- `reg_alpha`: regularização L1.
- `reg_lambda`: regularização L2.

## LightGBM

- `n_estimators`: número de árvores.
- `num_leaves`: complexidade estrutural.
- `max_depth`: limite opcional de profundidade.
- `learning_rate`: contribuição incremental.
- `min_child_samples`: mínimo de registros em folhas.
- `subsample`: amostragem de linhas.
- `colsample_bytree`: amostragem de colunas.
- `reg_alpha` / `reg_lambda`: regularizações.


## Gradient Boosting Regressor

- `n_estimators`: quantidade de árvores sequenciais.
- `learning_rate`: contribuição de cada nova árvore.
- `max_depth`: profundidade das árvores base.
- `min_samples_split`: mínimo de amostras para abrir uma divisão.
- `min_samples_leaf`: mínimo de amostras sustentando uma folha.
- `subsample`: proporção dos dados usada em cada estágio.

Interpretação de negócio: controla quanto o modelo aprende gradualmente com os erros anteriores. Modelos muito profundos ou com muitas árvores podem memorizar ruído.

## CatBoost

- `iterations`: quantidade de árvores.
- `depth`: profundidade das árvores.
- `learning_rate`: contribuição de cada árvore.
- `l2_leaf_reg`: regularização L2.
- `loss_function`: função de erro.
- `random_strength`: aleatoriedade nas divisões.
- `bagging_temperature`: intensidade do bagging bayesiano.
- `verbose`: controlar logs de treinamento.

Interpretação de negócio: CatBoost é especialmente útil quando há variáveis categóricas relevantes. A arquitetura deve permitir usar categorias nativas quando o pipeline configurado assim determinar.
