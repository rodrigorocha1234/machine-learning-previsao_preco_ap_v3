# Spec — Modelos e hiperparâmetros

## Regressão linear / múltipla
Equação:
`ŷ = β0 + β1x1 + ... + βpxp`

“Linear” e “múltipla” usam a mesma família matemática; a diferença é quantidade de preditores. Mantemos ambas como configurações para fins didáticos/negócio.

Parâmetros relevantes:
- `fit_intercept`
- `positive`

Interpretação: mantendo as demais variáveis constantes, `βj` representa a variação esperada no preço para uma unidade adicional de `xj`. Para dummies de `Zona`, o coeficiente é a diferença frente à categoria de referência.

## Polinomial
`ŷ = β0 + Σ βj φj(x)`, onde `φ` contém potências e interações.

Grid:
- `grau`: [2, 3]
- `include_bias`: [False]
- regressão final pode ser Linear/Ridge.

Interpretação: efeitos passam a depender do nível das variáveis; usar efeitos marginais e curvas parciais.

## Ridge
Objetivo:
`min ||y-Xβ||² + α||β||²₂`

Grid: `alpha` em escala log.

Negócio: reduz coeficientes extremos e tende a estabilizar estimativas quando variáveis são correlacionadas.

## Lasso
`min (1/(2n))||y-Xβ||² + α||β||₁`

Grid:
- `alpha`
- `max_iter`

Negócio: pode zerar coeficientes, produzindo seleção esparsa.

## Elastic Net
`min (1/(2n))||y-Xβ||² + α*l1_ratio||β||₁ + 0.5*α*(1-l1_ratio)||β||²₂`

Grid:
- `alpha`
- `l1_ratio`
- `max_iter`

## DecisionTreeRegressor
Predição por região terminal:
`ŷ(x) = média(y_i)` para amostras na folha alcançada.

Grid:
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`
- `ccp_alpha`

## RandomForestRegressor
`ŷ(x) = (1/B) Σ_b T_b(x)`

Grid:
- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`
- `bootstrap`

## SVR
`f(x)=Σ_i(α_i-α_i*)K(x_i,x)+b`

Grid:
- `kernel`: rbf, linear, poly
- `C`
- `epsilon`
- `gamma`
- `degree` quando poly

## MLPRegressor
`ŷ = W_L σ(...σ(W_1x+b_1)...)+b_L`

Grid:
- `hidden_layer_sizes`
- `activation`
- `alpha`
- `learning_rate_init`
- `early_stopping`

## XGBoost
Modelo aditivo:
`ŷ = Σ_t f_t(x)`, com `f_t` árvores adicionadas sequencialmente.

Grid:
- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `min_child_weight`
- `reg_alpha`
- `reg_lambda`

## LightGBM
Também é boosting de árvores:
`ŷ = Σ_t f_t(x)`

Grid:
- `n_estimators`
- `num_leaves`
- `max_depth`
- `learning_rate`
- `min_child_samples`
- `subsample`
- `colsample_bytree`
- `reg_alpha`
- `reg_lambda`

## Regra de interpretação
Para modelos sem uma equação global simples (árvores, florestas, boosting), a documentação deve registrar:
- fórmula estrutural;
- importâncias;
- permutation importance;
- SHAP opcional;
- exemplos de efeitos parciais.
