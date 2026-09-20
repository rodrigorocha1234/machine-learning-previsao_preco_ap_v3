# Tipagem Python sem `Any`

## Regra obrigatória

Não utilizar:

```python
from typing import Any
```

nem:

```python
Any
```

em assinaturas, atributos, Protocols, aliases ou estruturas internas da aplicação.

## Objetivo

A tipagem deve permanecer explícita, verificável e útil para:

- IDE;
- mypy/pyright;
- manutenção;
- refatoração;
- contratos entre módulos;
- redução de erros em runtime.

## Estratégias permitidas

### Tipos concretos

Prefira:

```python
pd.DataFrame
pd.Series
np.ndarray
dict[str, float]
dict[str, int | float | str]
list[str]
tuple[np.ndarray, np.ndarray]
Path
BaseEstimator
RegressorMixin
Figure
```

### Unions específicas

Quando existirem múltiplos tipos legítimos:

```python
str | Path
int | float
pd.DataFrame | np.ndarray
```

### `TypeVar` e `Generic`

Use quando uma abstração precisar preservar o tipo recebido.

Exemplo:

```python
from typing import Generic, TypeVar

TModelo = TypeVar("TModelo", bound=BaseEstimator)

class ResultadoModelo(Generic[TModelo]):
    def __init__(self, modelo: TModelo):
        self.modelo = modelo
```

### `Protocol`

Use para contratos estruturais.

Exemplo:

```python
from typing import Protocol

class ModeloRegressorProtocol(Protocol):
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "ModeloRegressorProtocol":
        ...

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        ...
```

### TypedDict

Use para dicionários com estrutura conhecida.

Exemplo:

```python
from typing import TypedDict

class MetricasRegressao(TypedDict):
    rmse: float
    mae: float
    mse: float
    r2: float
```

### dataclass

Prefira `dataclass` quando houver estrutura de dados rica.

Exemplo:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ResultadoFriedman:
    estatistica: float
    p_valor: float
    alpha: float
    significativo: bool
```

## Observer

Não usar:

```python
def atualizar(self, evento: str, dados: dict[str, Any]) -> None:
    ...
```

Prefira eventos tipados ou payloads específicos.

Exemplo:

```python
class EventoGridSearch(TypedDict):
    nome_modelo: str
    melhores_parametros: dict[str, int | float | str | bool | None]
    melhor_score: float
```

Ou crie dataclasses por tipo de evento.

## Configurações

Não usar:

```python
dict[str, Any]
```

Criar dataclasses ou TypedDicts específicos.

## Critério de aceite

A base de código não deve conter ocorrências de:

```text
typing.Any
Any
dict[str, Any]
list[Any]
tuple[Any, ...]
```

salvo texto documental explicando a proibição.


## Enums e tipagem

Use `StrEnum` quando um campo aceitar apenas um conjunto finito de valores.

Isso evita `str` genérico em contratos como:

- modelo;
- evento;
- scaler;
- ensemble;
- nível de drift;
- métrica;
- origem do peso.


## Regra de `Generic` e `TypeVar`

Use `Generic` e `TypeVar` quando for necessário preservar tipos entre entrada e saída.

Prefira `Protocol` como `bound` quando o contrato for comportamental.

Exemplo:

```python
TModelo = TypeVar(
    "TModelo",
    bound=RegressorProtocol,
)
```

Não use `TypeVar` sem restrição apenas para substituir `Any`.
