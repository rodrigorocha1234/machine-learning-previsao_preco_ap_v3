# MLflow como Observer

## Objetivo

Desacoplar as classes de Machine Learning do MLflow.

As classes de domínio geram dados e eventos. O `ObservadorMLflow` traduz esses eventos para operações do MLflow.

## Tipo de evento

Usar obrigatoriamente:

```python
from imobiliaria_ml.enums_pkg.tipo_evento import TipoEvento
```

Não usar strings livres para eventos.

## Interfaces

```python
class Observador(ABC):
    @abstractmethod
    def atualizar(
        self,
        evento: TipoEvento,
        dados: EventoMlflowProtocol,
    ) -> None:
        ...
```

O payload deve ser tipado com `TypedDict`, `Protocol` ou `dataclass`, nunca com `Any`.

```python
class SujeitoObservavel:
    def __init__(self):
        self._observadores: list[Observador] = []

    def adicionar_observador(
        self,
        observador: Observador
    ) -> None:
        self._observadores.append(observador)

    def notificar(
        self,
        evento: TipoEvento,
        dados: EventoMlflowProtocol,
    ) -> None:
        for observador in self._observadores:
            observador.atualizar(evento, dados)
```

## Eventos mínimos

- `TipoEvento.EDA_FINALIZADA`
- `TipoEvento.GRIDSEARCH_INICIADO`
- `TipoEvento.GRIDSEARCH_FINALIZADO`
- `TipoEvento.VALIDACAO_FINALIZADA`
- `TipoEvento.FRIEDMAN_FINALIZADO`
- `TipoEvento.NEMENYI_FINALIZADO`
- `TipoEvento.ENSEMBLE_FINALIZADO`
- `TipoEvento.MODELO_CAMPEAO`
- `TipoEvento.METRICAS_NEGOCIO_FINALIZADAS`
- `TipoEvento.DRIFT_DETECTADO`
- `TipoEvento.CURVA_APRENDIZADO_GERADA`

## Registro direto no MLflow

O Observer deve escolher a API adequada:

- `mlflow.log_params`
- `mlflow.log_metrics`
- `mlflow.log_dict`
- `mlflow.log_text`
- `mlflow.log_table`
- `mlflow.log_figure`
- `mlflow.pyfunc.log_model`

Arquivos temporários são exceção e devem ser removidos após o registro.

## Tracking URI

A aplicação deve ler:

```text
MLFLOW_TRACKING_URI
```

apontando para o serviço do MLflow levantado pelo `docker-compose.yml` na raiz.

Não duplicar servidor MLflow dentro da aplicação.
