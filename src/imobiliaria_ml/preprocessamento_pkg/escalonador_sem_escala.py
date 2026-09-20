"""Estratégia de manutenção da escala original sem transformação."""

from sklearn.preprocessing import FunctionTransformer
from .estrategia_escalonamento import EstrategiaEscalonamento


class EscalonadorSemEscala(EstrategiaEscalonamento):
    """Transformador de identidade para modelos que dispensam escalonamento."""

    def criar_transformador(self) -> FunctionTransformer:
        """Retorna transformador de identidade."""
        return FunctionTransformer(func=None, validate=False)
