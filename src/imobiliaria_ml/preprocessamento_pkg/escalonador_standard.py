"""Estratégia de escalonamento padrão com média 0 e variância unitária."""

from sklearn.preprocessing import StandardScaler
from .estrategia_escalonamento import EstrategiaEscalonamento


class EscalonadorStandard(EstrategiaEscalonamento):
    """Escalonador baseado em StandardScaler."""

    def criar_transformador(self) -> StandardScaler:
        """Retorna instância de StandardScaler."""
        return StandardScaler()
