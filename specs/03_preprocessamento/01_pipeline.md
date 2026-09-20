# Pipeline de Pré-processamento

Usar `ColumnTransformer` + `Pipeline` para impedir leakage.

## Numéricas

Pipeline padrão:

```text
SimpleImputer(strategy="median")
→ escalonador configurado para o modelo
```

## Categóricas

```text
SimpleImputer(strategy="most_frequent")
→ OneHotEncoder(handle_unknown="ignore")
```

## Escalonamento por família

| Modelo | Escalonador padrão |
|---|---|
| Linear / Múltipla | StandardScaler opcional, recomendado para comparação de coeficientes padronizados |
| Polinomial | StandardScaler depois de PolynomialFeatures |
| Ridge | StandardScaler |
| Lasso | StandardScaler |
| Elastic Net | StandardScaler |
| SVR | StandardScaler |
| Rede Neural | StandardScaler ou MinMaxScaler |
| Árvore | sem escala |
| Random Forest | sem escala |
| XGBoost | sem escala |
| LightGBM | sem escala |

`RobustScaler` é alternativa quando a EDA indicar outliers fortes.

A escolha do scaler é uma Strategy.


## Enum de escalonamento

A seleção de scaler deve usar `TipoEscalonador`.

Exemplo:

```python
def criar_escalonador(
    self,
    tipo: TipoEscalonador
) -> TransformerMixin:
    ...
```
