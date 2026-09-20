"""Interface base abstrata para estratégias de ensemble."""

from abc import ABC, abstractmethod
from ..modelos_pkg.regressor_protocol import RegressorProtocol


class EstrategiaEnsemble(ABC):
    """Classe base para estratégias de composição de múltiplos modelos."""

    @abstractmethod
    def criar_estimador(
        self,
        estimadores_elegiveis: list[tuple[str, RegressorProtocol]],
    ) -> RegressorProtocol:
        """Instancia o ensemble concreto configurado com os estimadores fornecidos.

        Parameters
        ----------
        estimadores_elegiveis : list[tuple[str, RegressorProtocol]]
            Lista de tuplas (nome, estimador) dos modelos estatisticamente elegíveis.

        Returns
        -------
        RegressorProtocol
            Instância do ensemble configurado.
        """
        ...
