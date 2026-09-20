"""Estrutura concreta para encapsulamento de carga de eventos do Observer."""

from dataclasses import dataclass, field
from matplotlib.figure import Figure
import pandas as pd
from .evento_mlflow_protocol import EventoMlflowProtocol


@dataclass(frozen=True)
class CargaEvento(EventoMlflowProtocol):
    """Carga útil de eventos trafegados para o MLflow sem uso de Any."""

    valores: dict[
        str,
        str
        | int
        | float
        | bool
        | pd.DataFrame
        | Figure
        | dict[str, float | int | str | bool | None]
        | list[str]
        | None,
    ] = field(default_factory=dict)

    def para_dicionario(
        self,
    ) -> dict[
        str,
        str
        | int
        | float
        | bool
        | pd.DataFrame
        | Figure
        | dict[str, float | int | str | bool | None]
        | list[str]
        | None,
    ]:
        """Retorna o dicionário de valores encapsulados."""
        return dict(self.valores)
