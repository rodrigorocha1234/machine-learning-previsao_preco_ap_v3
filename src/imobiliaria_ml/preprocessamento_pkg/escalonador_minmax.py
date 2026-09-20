"""Estratégia de escalonamento linear no intervalo [0, 1]."""

from sklearn.preprocessing import MinMaxScaler
from .estrategia_escalonamento import EstrategiaEscalonamento


class EscalonadorMinmax(EstrategiaEscalonamento):
    """Escalonador baseado em MinMaxScaler."""

    def criar_transformador(self) -> MinMaxScaler:
        """Retorna instância de MinMaxScaler."""
        return MinMaxScaler()
