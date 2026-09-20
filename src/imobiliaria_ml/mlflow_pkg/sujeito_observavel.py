"""Sujeito observável que gerencia e notifica observadores cadastrados."""

from ..enums_pkg.tipo_evento import TipoEvento
from .observador import Observador
from .evento_mlflow_protocol import EventoMlflowProtocol


class SujeitoObservavel:
    """Implementação do padrão GoF Observer (Subject)."""

    def __init__(self) -> None:
        self._observadores: list[Observador] = []

    def adicionar_observador(self, observador: Observador) -> None:
        """Registra um novo observador no sujeito."""
        if observador not in self._observadores:
            self._observadores.append(observador)

    def remover_observador(self, observador: Observador) -> None:
        """Remove um observador previamente registrado."""
        if observador in self._observadores:
            self._observadores.remove(observador)

    def notificar(
        self,
        evento: TipoEvento,
        dados: EventoMlflowProtocol,
    ) -> None:
        """Notifica todos os observadores sobre a ocorrência de um evento."""
        for observador in self._observadores:
            observador.atualizar(evento, dados)
