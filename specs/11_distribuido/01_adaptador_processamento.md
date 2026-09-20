# Extensão para Processamento Distribuído

## Objetivo

Não acoplar a aplicação ao scikit-learn local.

## Contrato tipado

Use `Generic` e `TypeVar` para preservar os tipos.

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
    @abstractmethod
    def ajustar(
        self,
        modelo: TModelo,
        dados: TDados,
        alvo: TAlvo,
    ) -> TModelo:
        ...

    @abstractmethod
    def prever(
        self,
        modelo: TModelo,
        dados: TDados,
    ) -> np.ndarray:
        ...
```

Implementação atual:

```text
ExecutorSklearn
```

Implementação futura:

```text
AdaptadorDistribuido
```

## Adapter tipado

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

O Adapter poderá envolver Spark MLlib, Dask-ML, Ray ou outro backend.

## Restrição

Nem todos os algoritmos, hiperparâmetros e objetos sklearn possuem equivalentes distribuídos idênticos.

O adapter deve declarar capacidades suportadas e diferenças de semântica.

Não usar `Any`.
