"""Protocolo estrutural para transporte tipado de dados de eventos do MLflow."""

from typing import Protocol
from matplotlib.figure import Figure
import pandas as pd


class EventoMlflowProtocol(Protocol):
    """Contrato estrutural para payloads de eventos sem recorrer a Any."""

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
        """Converte as informações do evento para um dicionário com tipos explícitos."""
        ...
