# Nomenclatura dos Packages Python

## Regra obrigatória

Todos os diretórios Python localizados dentro de:

```text
src/imobiliaria_ml/
```

devem terminar com o sufixo:

```text
_pkg
```

Exemplos:

```text
enums_pkg/
carregamento_pkg/
qualidade_pkg/
eda_pkg/
preprocessamento_pkg/
modelos_pkg/
selecao_pkg/
validacao_pkg/
ensemble_pkg/
explicabilidade_pkg/
negocio_pkg/
drift_pkg/
mlflow_pkg/
distribuido_pkg/
pipeline_pkg/
aplicacao_pkg/
```

## Objetivo

A convenção reduz o risco de colisão com:

- palavras reservadas do Python;
- módulos da biblioteca padrão;
- bibliotecas de terceiros;
- nomes genéricos ou ambíguos;
- módulos futuros com nomes semelhantes.

## Regra para arquivos

A regra permanece:

```text
1 classe = 1 arquivo .py
```

O sufixo `_pkg` vale para diretórios/packages, não para arquivos `.py`.

Exemplo:

```text
src/imobiliaria_ml/modelos_pkg/
├── __init__.py
├── estrategia_modelo.py
├── regressao_linear.py
├── regressao_random_forest.py
└── fabrica_modelos.py
```

## Imports

Exemplo:

```python
from imobiliaria_ml.modelos_pkg.regressao_random_forest import (
    RegressaoRandomForest,
)

from imobiliaria_ml.ensemble_pkg.votacao_regressor import (
    VotacaoRegressor,
)

from imobiliaria_ml.validacao_pkg.teste_friedman import (
    TesteFriedman,
)

from imobiliaria_ml.mlflow_pkg.observador_mlflow import (
    ObservadorMLflow,
)
```

## Proibido

Evitar packages internos como:

```text
modelos/
ensemble/
validacao/
mlflow/
pipeline/
```

sem o sufixo `_pkg`.
