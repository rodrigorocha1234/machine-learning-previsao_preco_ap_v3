import time

from app_build.observabilidade_metricas.emissor_logs import EmissorLogs, EventoLog
from app_build.observabilidade_metricas.servico_telemetria import ServicoTelemetria
from app_build.orquestracao_pipeline.contexto_execucao import ContextoExecucao
from app_build.orquestracao_pipeline.executor_esteira import ExecutorEsteira
from app_build.rastreamento_mlflow.observador_mlflow import ObservadorMlflow


class FluxoPrincipal:
    def __init__(self) -> None:
        self._emissor = EmissorLogs()
        self._telemetria = ServicoTelemetria(porta=8000)
        self._observador_mlflow = ObservadorMlflow(
            coletor=self._telemetria.coletor,
        )
        self._executor = ExecutorEsteira(
            emissor_logs=self._emissor,
            coletor=self._telemetria.coletor,
        )

    def iniciar(self) -> ContextoExecucao:
        msg_inicio = (
            "=== Iniciando Pipeline de Previsao de Precos de Apartamentos"
            " (Ribeirao Preto/SP) ==="
        )
        print(msg_inicio)
        self._emissor.emitir_evento(
            EventoLog(nivel="INFO", servico="executor_esteira", mensagem=msg_inicio)
        )
        self._telemetria.iniciar_servidor()
        self._observador_mlflow.inicializar()

        contexto = ContextoExecucao(coletor=self._telemetria.coletor)
        contexto.despachante.registrar_observador(self._observador_mlflow)

        ts_inicio_pipeline = time.perf_counter()
        contexto_final = self._executor.executar_esteira(contexto)
        duracao_total = time.perf_counter() - ts_inicio_pipeline

        # Registra conclusão do pipeline com duração total
        self._telemetria.coletor.registrar_pipeline_concluido(
            duracao_total_segundos=duracao_total,
            total_etapas=20,
            etapas_com_falha=0,
        )

        msg_fim = (
            f"=== Pipeline Executado com Sucesso! "
            f"[Duracao total: {duracao_total:.1f}s] ==="
        )
        print(msg_fim)
        self._emissor.emitir_evento(
            EventoLog(nivel="INFO", servico="executor_esteira", mensagem=msg_fim)
        )
        return contexto_final


def executar_pipeline() -> None:
    fluxo = FluxoPrincipal()
    fluxo.iniciar()


if __name__ == "__main__":
    executar_pipeline()
