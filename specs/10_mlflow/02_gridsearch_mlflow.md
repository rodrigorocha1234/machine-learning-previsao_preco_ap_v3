# Registro do Grid Search no MLflow

## Regra

Todos os resultados relevantes do Grid Search devem ser registrados diretamente no MLflow.

Evitar gerar arquivos permanentes no disco local.

## Registro recomendado

Para estimadores sklearn, utilizar `mlflow.sklearn.autolog()` quando apropriado.

Além do autolog, o `ObservadorMLflow` deve padronizar o registro entre sklearn, XGBoost e LightGBM.

Registrar:

- grade completa;
- melhor combinação;
- `best_score_`;
- `best_params_`;
- quantidade de combinações;
- tempo total da busca;
- `cv_results_`;
- ranking;
- melhor estimador.

## Exemplo

```python
resultados = pd.DataFrame(busca.cv_results_)

mlflow.log_table(
    resultados,
    "gridsearch/cv_results.json"
)

mlflow.log_dict(
    busca.best_params_,
    "gridsearch/melhores_parametros.json"
)
```

## Runs

Estratégia recomendada:

```text
run pai
└── Grid Search do modelo
    ├── child run combinação 1
    ├── child run combinação 2
    └── ...
```

A política de retenção de child runs deve ser configurável quando a grade for muito grande.
