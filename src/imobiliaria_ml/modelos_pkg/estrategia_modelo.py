"""Classe abstrata genérica para estratégias de modelos de regressão."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from ..enums_pkg.tipo_modelo import TipoModelo
from ..enums_pkg.tipo_escalonador import TipoEscalonador
from .regressor_protocol import RegressorProtocol
from .parametros_modelo import ParametrosModelo

TModelo = TypeVar("TModelo", bound=RegressorProtocol)


class EstrategiaModelo(Generic[TModelo], ABC):
    """Interface base genérica para estimadores de Machine Learning."""

    @abstractmethod
    def criar_modelo(self, parametros: ParametrosModelo | None = None) -> TModelo:
        """Instancia o modelo concreto preservando o tipo TModelo."""
        ...

    @abstractmethod
    def obter_grade_hiperparametros(
        self,
    ) -> dict[str, list[int | float | str | bool | tuple[int, ...] | None]]:
        """Retorna a grade inicial de hiperparâmetros para o GridSearch."""
        ...

    @abstractmethod
    def obter_tipo_modelo(self) -> TipoModelo:
        """Retorna o enum TipoModelo correspondente."""
        ...

    @abstractmethod
    def obter_escalonador_padrao(self) -> TipoEscalonador:
        """Retorna a estratégia de escalonamento recomendada para o modelo."""
        ...

    def obter_nome(self) -> str:
        """Retorna o identificador textual do modelo."""
        return str(self.obter_tipo_modelo().value)
