# Política de Persistência de Artefatos

## Regra principal

O MLflow é o repositório oficial dos artefatos experimentais.

Os artefatos produzidos durante:

- EDA;
- Grid Search;
- validação cruzada;
- Friedman;
- Nemenyi;
- seleção do campeão;
- ensemble;
- explicabilidade;
- métricas da imobiliária;
- drift;
- diagnóstico de underfitting/overfitting;

não devem ser persistidos permanentemente no filesystem da aplicação antes do envio ao MLflow.

## Fluxo desejado

```text
objeto em memória
      ↓
evento da aplicação
      ↓
ObservadorMLflow
      ↓
MLflow Tracking / Artifact Store
```

## APIs preferenciais

### Dicionários

```python
mlflow.log_dict(
    dados,
    "estatistica/friedman.json"
)
```

### Texto

```python
mlflow.log_text(
    texto,
    "explicabilidade/equacao_modelo.txt"
)
```

### Tabelas

```python
mlflow.log_table(
    dataframe,
    "gridsearch/cv_results.json"
)
```

### Figuras

```python
mlflow.log_figure(
    figura,
    "validacao/curva_aprendizado.png"
)
```

### Métricas simples

```python
mlflow.log_metrics({
    "rmse": rmse,
    "mae": mae,
    "r2": r2
})
```

### Parâmetros simples

```python
mlflow.log_params({
    "max_depth": 12,
    "n_estimators": 500
})
```

## Regra para arquivos temporários

Arquivos locais temporários são permitidos somente quando:

1. uma biblioteca exigir obrigatoriamente um caminho físico;
2. o formato não puder ser registrado diretamente por uma API do MLflow;
3. o arquivo for removido imediatamente após `mlflow.log_artifact()`.

Usar preferencialmente:

```python
tempfile.TemporaryDirectory()
```

ou:

```python
tempfile.NamedTemporaryFile()
```

## Proibido

Não manter como fluxo padrão diretórios persistentes do tipo:

```text
artefatos/
gridsearch/
eda_pkg/
validacao_pkg/
```

apenas para posterior upload ao MLflow.

## Responsabilidade do Observer

As classes de domínio devem enviar os objetos em memória ao `ObservadorMLflow`.

Exemplo:

```python
sujeito.notificar(
    "gridsearch_finalizado",
    {
        "resultados": resultados_grid,
        "melhores_parametros": melhores_parametros,
    }
)
```

O `ObservadorMLflow` decide como registrar cada objeto.
