from typing import Protocol, runtime_checkable

from app_build.orquestracao_pipeline.contexto_execucao import ContextoExecucao


@runtime_checkable
class ContratoEtapa(Protocol):
    @property
    def nome_etapa(self) -> str: ...

    def executar(self, contexto: ContextoExecucao) -> None: ...
