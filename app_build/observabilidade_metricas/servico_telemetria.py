from typing import Final

from prometheus_client import start_http_server

from app_build.observabilidade_metricas.coletor_prometheus import ColetorPrometheus


class ServicoTelemetria:
    def __init__(
        self, porta: int = 8000, coletor: ColetorPrometheus | None = None
    ) -> None:
        self._porta: Final[int] = porta
        self._coletor: Final[ColetorPrometheus] = coletor or ColetorPrometheus()

    def iniciar_servidor(self) -> None:
        try:
            start_http_server(self._porta)
        except OSError:
            pass

    @property
    def coletor(self) -> ColetorPrometheus:
        return self._coletor
