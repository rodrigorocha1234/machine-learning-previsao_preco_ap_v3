"""Estratégia de escalonamento robusto a outliers baseado em mediana e IQR."""

from sklearn.preprocessing import RobustScaler
from .estrategia_escalonamento import EstrategiaEscalonamento


class EscalonadorRobusto(EstrategiaEscalonamento):
    """Escalonador baseado em RobustScaler."""

    def criar_transformador(self) -> RobustScaler:
        """Retorna instância de RobustScaler."""
        return RobustScaler()
