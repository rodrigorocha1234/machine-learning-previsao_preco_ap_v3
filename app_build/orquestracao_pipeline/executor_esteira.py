import time
from collections.abc import Callable, Sequence
from typing import Final

from app_build.observabilidade_metricas.coletor_prometheus import ColetorPrometheus
from app_build.observabilidade_metricas.emissor_logs import EmissorLogs, EventoLog
from app_build.orquestracao_pipeline.contexto_execucao import ContextoExecucao
from app_build.orquestracao_pipeline.contrato_etapa import ContratoEtapa
from app_build.orquestracao_pipeline.fabrica_etapas import (
    Etapa01CarregarConfiguracoes,
    Etapa02ValidarConfiguracoes,
    Etapa03CarregarDados,
    Etapa04ValidarDados,
    Etapa05Staging,
    Etapa06SepararHoldout,
    Etapa07BloquearHoldout,
    Etapa08Eda,
    Etapa09Drift,
    Etapa10NestedCv,
    Etapa11Estatistica,
    Etapa12SelecaoModelo,
    Etapa13TuningFinal,
    Etapa14TreinamentoFinal,
    Etapa15CongelarConfiguracao,
    Etapa16AbrirHoldout,
    Etapa17AvaliacaoHoldout,
    Etapa18RegrasNegocio,
    Etapa19RastreamentoMlflow,
    Etapa20Serving,
)


class ExecutorEsteira:
    def __init__(
        self,
        etapas: Sequence[ContratoEtapa] | None = None,
        emissor_logs: EmissorLogs | None = None,
        coletor: ColetorPrometheus | None = None,
        persistir_metricas: Callable[[], None] | None = None,
    ) -> None:
        padrao_etapas: Final[tuple[ContratoEtapa, ...]] = (
            Etapa01CarregarConfiguracoes(),
            Etapa02ValidarConfiguracoes(),
            Etapa03CarregarDados(),
            Etapa04ValidarDados(),
            Etapa05Staging(),
            Etapa06SepararHoldout(),
            Etapa07BloquearHoldout(),
            Etapa08Eda(),
            Etapa09Drift(),
            Etapa10NestedCv(),
            Etapa11Estatistica(),
            Etapa12SelecaoModelo(),
            Etapa13TuningFinal(),
            Etapa14TreinamentoFinal(),
            Etapa15CongelarConfiguracao(),
            Etapa16AbrirHoldout(),
            Etapa17AvaliacaoHoldout(),
            Etapa18RegrasNegocio(),
            Etapa19RastreamentoMlflow(),
            Etapa20Serving(),
        )
        self._etapas: Final[tuple[ContratoEtapa, ...]] = tuple(etapas or padrao_etapas)
        self._emissor: Final[EmissorLogs] = emissor_logs or EmissorLogs()
        self._coletor: Final[ColetorPrometheus | None] = coletor
        self._persistir_metricas = persistir_metricas or (lambda: None)

    def executar_esteira(
        self, contexto: ContextoExecucao | None = None
    ) -> ContextoExecucao:
        ctx = contexto or ContextoExecucao()
        total_etapas = len(self._etapas)

        for indice, etapa in enumerate(self._etapas, start=1):
            msg_inicio = (
                f">>> Executando etapa: {etapa.nome_etapa} "
                f"(Etapa {indice} de {total_etapas})"
            )
            print(msg_inicio)
            self._emissor.emitir_evento(
                EventoLog(nivel="INFO", servico="executor_esteira", mensagem=msg_inicio)
            )

            if self._coletor is not None:
                self._coletor.registrar_inicio_etapa(
                    nome_etapa=etapa.nome_etapa, indice=indice, total=total_etapas
                )

            self._persistir_metricas()
            ts_inicio = time.perf_counter()
            sucesso = True
            try:
                etapa.executar(ctx)
            except Exception:
                sucesso = False
                raise
            finally:
                duracao = time.perf_counter() - ts_inicio
                msg_fim = (
                    f"--- Concluida etapa: {etapa.nome_etapa} "
                    f"(Etapa {indice} de {total_etapas}) [{duracao:.2f}s]"
                )
                print(msg_fim)
                self._emissor.emitir_evento(
                    EventoLog(
                        nivel="INFO" if sucesso else "ERROR",
                        servico="executor_esteira",
                        mensagem=msg_fim,
                    )
                )
                if self._coletor is not None:
                    self._coletor.registrar_fim_etapa(
                        nome_etapa=etapa.nome_etapa,
                        indice=indice,
                        total=total_etapas,
                        duracao_segundos=duracao,
                        sucesso=sucesso,
                    )
                self._persistir_metricas()

        return ctx
