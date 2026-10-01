"""Recalcula a avaliação configurada e publica métricas sem promover um modelo."""

import time

from app_build.observabilidade_metricas.servico_telemetria import ServicoTelemetria
from app_build.orquestracao_pipeline import fabrica_etapas as etapas
from app_build.orquestracao_pipeline.contexto_execucao import ContextoExecucao
from app_build.orquestracao_pipeline.executor_esteira import ExecutorEsteira
from app_build.rastreamento_mlflow.observador_mlflow import ObservadorMlflow


def recalcular_metricas() -> None:
    telemetria = ServicoTelemetria()
    observador = ObservadorMlflow(coletor=telemetria.coletor)
    observador.inicializar()
    contexto = ContextoExecucao(coletor=telemetria.coletor)
    contexto.despachante.registrar_observador(observador)
    etapas_avaliacao = (
        etapas.Etapa01CarregarConfiguracoes(),
        etapas.Etapa02ValidarConfiguracoes(),
        etapas.Etapa03CarregarDados(),
        etapas.Etapa04ValidarDados(),
        etapas.Etapa05Staging(),
        etapas.Etapa06SepararHoldout(),
        etapas.Etapa07BloquearHoldout(),
        etapas.Etapa08Eda(),
        etapas.Etapa09Drift(),
        etapas.Etapa10NestedCv(),
        etapas.Etapa11Estatistica(),
        etapas.Etapa12SelecaoModelo(),
        etapas.Etapa13TuningFinal(),
        etapas.Etapa14TreinamentoFinal(),
        etapas.Etapa15CongelarConfiguracao(),
        etapas.Etapa16AbrirHoldout(),
        etapas.Etapa17AvaliacaoHoldout(),
        etapas.Etapa18RegrasNegocio(),
    )
    inicio = time.perf_counter()
    ExecutorEsteira(
        etapas=etapas_avaliacao,
        coletor=telemetria.coletor,
        persistir_metricas=telemetria.salvar_metricas,
    ).executar_esteira(contexto)
    telemetria.coletor.registrar_pipeline_concluido(
        duracao_total_segundos=time.perf_counter() - inicio,
        total_etapas=len(etapas_avaliacao),
        etapas_com_falha=0,
    )
    telemetria.salvar_metricas()
    print("Avaliação concluída e métricas persistidas. Modelo publicado preservado.")


if __name__ == "__main__":
    recalcular_metricas()
