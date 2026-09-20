# Equações e Interpretação

## Regressão Linear / Múltipla

\[
\hat{y} = \beta_0 + \beta_1x_1 + \cdots + \beta_px_p
\]

Exemplo:

```text
Valor = 80.000 + 45.000*Quartos + 1.500*Metragem + ...
```

Interpretação: mantendo as demais variáveis constantes, `β_Metragem = 1.500` indica acréscimo médio de R$ 1.500 por unidade adicional de metragem, na escala em que o modelo foi ajustado.

Quando houver `StandardScaler`, registrar também coeficientes convertidos para uma interpretação utilizável no negócio.

## Polinomial

\[
\hat{y} = \beta_0 + \beta_1x + \beta_2x^2 + \cdots
\]

Permite efeitos não lineares. A interpretação deve considerar o efeito marginal, não apenas um coeficiente isolado.

## Ridge

\[
\min_{\beta}\sum_i(y_i-\hat y_i)^2+\alpha\sum_j\beta_j^2
\]

Interpretação: reduz coeficientes para controlar variância, normalmente mantendo todas as variáveis.

## Lasso

\[
\min_{\beta}\sum_i(y_i-\hat y_i)^2+\alpha\sum_j|\beta_j|
\]

Pode zerar coeficientes e atuar como seleção de variáveis.

## Elastic Net

\[
\min_{\beta}\sum_i(y_i-\hat y_i)^2+
\alpha[\rho\sum_j|\beta_j|+(1-\rho)\sum_j\beta_j^2]
\]

Combina L1 e L2.

## Árvore

Modelo por regras:

```text
se Zona = Centro e Metragem > 120 → ramo A
senão → ramo B
```

Não existe um único vetor global de coeficientes.

## Random Forest

\[
\hat y = \frac{1}{T}\sum_{t=1}^{T} f_t(x)
\]

É a média das árvores.

## SVR

\[
f(x)=\sum_i(\alpha_i-\alpha_i^*)K(x_i,x)+b
\]

A interpretação depende do kernel. Para RBF, não há coeficiente direto por atributo comparável à regressão linear.

## Rede Neural

\[
\hat y = f_L(W_L f_{L-1}(...f_1(W_1x+b_1)...)+b_L)
\]

Pesos internos não devem ser tratados como efeito de negócio direto.

## XGBoost / LightGBM

\[
\hat y = \sum_{m=1}^{M} \eta f_m(x)
\]

Predição é soma de árvores impulsionadas.

## Artefatos de explicabilidade

Para modelos lineares:

- `coeficientes.csv`
- `equacao_modelo.txt`
- `interpretacao_coeficientes.md`

Para modelos sem coeficientes globais úteis:

- importância por permutação;
- SHAP, se habilitado;
- PDP/ICE opcional;
- regras da árvore quando o modelo permitir.
