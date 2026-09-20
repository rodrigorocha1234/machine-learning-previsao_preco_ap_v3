# Spec — Pré-processamento por família de modelo

## Categóricas
`OneHotEncoder(handle_unknown="ignore")`.

Para famílias lineares, usar `drop="first"` quando a interpretação por categoria de referência for desejada.
Para árvores/boosting, `drop=None` é aceitável.

## Numéricas
### Regressão Linear, Múltipla, Polinomial
- imputação pela mediana;
- `StandardScaler` recomendado quando há polynomial features ou comparação de magnitude de coeficientes.

### Ridge, Lasso, Elastic Net
- `StandardScaler` obrigatório/recomendado, pois a regularização depende da escala.

### SVR
- `StandardScaler` obrigatório na prática.

### MLPRegressor
- `StandardScaler` obrigatório/recomendado.

### Árvore de Decisão, Random Forest, XGBoost, LightGBM
- escalonamento não é necessário;
- imputação continua necessária;
- categóricas precisam de codificação no pipeline atual.

## Alternativas
A fábrica de escaladores deve suportar:
- `StandardScaler`
- `MinMaxScaler`
- `RobustScaler`
- `PowerTransformer`
- `QuantileTransformer` (opcional e configurável)

A escolha final é parte da estratégia/pipeline e pode entrar no GridSearch quando fizer sentido.
