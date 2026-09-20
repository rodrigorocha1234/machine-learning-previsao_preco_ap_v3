"""Interface abstrata para observadores de eventos do pipeline."""

from abc import ABC, abstractmethod
from ..enums_pkg.tipo_evento import TipoEvento
from .evento_mlflow_protocol import EventoMlflowProtocol


class Observador(ABC):
    """Contrato abstrato para recepção de notificações de eventos."""

    @abstractmethod
    def atualizar(
        self,
        evento: TipoEvento,
        dados: EventoMlflowProtocol,
    ) -> None:
        """Processa a notificação recebida do SujeitoObservavel.

        Parameters
        ----------
        evento : TipoEvento
            Identificador do evento ocorrido.
        dados : EventoMlflowProtocol
            Carga de dados associada ao evento.
        """
        ...
