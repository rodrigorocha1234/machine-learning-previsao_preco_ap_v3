"""Interface base para estratégias de escalonamento numérico."""

from abc import ABC, abstractmethod
from sklearn.base import TransformerMixin, BaseEstimator


class EstrategiaEscalonamento(ABC):
    """Classe base abstrata para escalonadores numéricos."""

    @abstractmethod
    def criar_transformador(self) -> TransformerMixin | BaseEstimator:
        """Cria e retorna uma instância do transformador de escalonamento."""
        ...
