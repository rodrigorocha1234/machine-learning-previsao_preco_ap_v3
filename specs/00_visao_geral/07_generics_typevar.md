# Regra de uso de `Generic` e `TypeVar`

## Objetivo

Usar `Generic` e `TypeVar` quando uma abstração precisar preservar a relação entre tipos de entrada, saída ou implementação concreta.

A finalidade é reforçar a tipagem sem recorrer a `Any`, mantendo reutilização e contratos fortes.

## Regra principal

Use `Generic` e `TypeVar` quando houver ganho real de precisão de tipo.

Não use generics apenas para tornar o código artificialmente complexo.

## Preferência de bound

Quando o contrato for comportamental, prefira um `Protocol` como bound do `TypeVar`.

Exemplo:

```python
from typing import Protocol, TypeVar

class RegressorProtocol(Protocol):
    def fit(
        self,
        X: DadosEntrada,
        y: AlvoEntrada,
    ) -> "RegressorProtocol":
        ...

    def predict(
        self,
        X: DadosEntrada,
    ) -> np.ndarray:
        ...

TRegressor = TypeVar(
    "TRegressor",
    bound=RegressorProtocol,
)
```

## Strategy de modelos

Use generic para preservar o tipo concreto do estimador:

```python
from typing import Generic, TypeVar

TModelo = TypeVar(
    "TModelo",
    bound=RegressorProtocol,
)

class EstrategiaModelo(Generic[TModelo]):

    def criar_modelo(
        self,
        parametros: ParametrosModelo,
    ) -> TModelo:
        ...
```

Implementação:

```python
class RegressaoRandomForest(
    EstrategiaModelo[RandomForestRegressor]
):

    def criar_modelo(
        self,
        parametros: ParametrosModelo,
    ) -> RandomForestRegressor:
        ...
```

## Resultado de treinamento

```python
from dataclasses import dataclass
from typing import Generic, TypeVar

TModelo = TypeVar(
    "TModelo",
    bound=RegressorProtocol,
)

@dataclass(frozen=True)
class ResultadoTreinamento(Generic[TModelo]):
    modelo: TModelo
    rmse: float
    mae: float
    r2: float
```

## Executor

Use generics quando o executor precisar preservar o tipo do modelo e dos dados.

```python
TModelo = TypeVar(
    "TModelo",
    bound=RegressorProtocol,
)

TDados = TypeVar(
    "TDados",
    pd.DataFrame,
    np.ndarray,
)

TAlvo = TypeVar(
    "TAlvo",
    pd.Series,
    np.ndarray,
)

class ExecutorML(
    Generic[TModelo, TDados, TAlvo]
):
    def ajustar(
        self,
        modelo: TModelo,
        dados: TDados,
        alvo: TAlvo,
    ) -> TModelo:
        ...
```

## Adapter distribuído

```python
TModeloOrigem = TypeVar(
    "TModeloOrigem",
    bound=RegressorProtocol,
)

TModeloDestino = TypeVar(
    "TModeloDestino",
)

class AdaptadorDistribuido(
    Generic[TModeloOrigem, TModeloDestino]
):

    def adaptar(
        self,
        modelo: TModeloOrigem,
    ) -> TModeloDestino:
        ...
```

Se `TModeloDestino` possuir contrato comportamental conhecido, crie também um `Protocol` específico e use como bound.

## Objetos de resultado

Use `Generic` em estruturas que encapsulem um modelo mantendo o tipo concreto:

- `ResultadoTreinamento[TModelo]`
- `ResultadoValidacao[TModelo]`
- `ModeloRegistrado[TModelo]`
- `ResultadoEnsemble[TModelo]`, quando aplicável.

## Factory heterogênea

Não force um único `Generic[TModelo]` em uma factory que cria tipos heterogêneos se isso reduzir a precisão.

Para factories heterogêneas, prefira:

- `Protocol`;
- overloads;
- unions específicas;
- tipo abstrato comum.

Não use `Any`.

## Critério de uso

Use `Generic` / `TypeVar` quando:

1. o mesmo tipo entra e deve sair preservado;
2. uma classe parametrizada precisa reter o tipo concreto;
3. um Adapter converte entre dois tipos relacionados;
4. o compilador/verificador pode obter benefício real.

Evite quando:

1. o tipo concreto não precisa ser preservado;
2. um `Protocol` simples já resolve;
3. a genericidade só adiciona ruído;
4. um enum ou dataclass resolve melhor o domínio.

## Compatibilidade com a regra sem `Any`

Generics devem substituir abstrações vagas, não esconder tipos desconhecidos.

Proibido:

```python
T = TypeVar("T")
```

usado apenas para contornar contratos mal definidos.

Prefira bounds, constraints ou Protocols quando houver um contrato conhecido.
