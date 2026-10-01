"""Serving nativo MLflow com telemetria HTTP e versão fixada na inicialização."""

import os

import mlflow
import uvicorn
from mlflow.pyfunc import scoring_server
from mlflow.tracking import MlflowClient

from app_build.observabilidade_metricas.adaptador_serving import AdaptadorServing
from app_build.observabilidade_metricas.metricas_serving import MetricasServing


def criar_aplicacao() -> AdaptadorServing:
    nome = os.environ.get("MODELO_NOME", "previsao_preco_apartamento_modelo")
    alias = os.environ.get("MODELO_ALIAS", "champion")
    cliente = MlflowClient()
    versao = cliente.get_model_version_by_alias(nome, alias)
    modelo = mlflow.pyfunc.load_model(f"models:/{nome}/{versao.version}")
    empacotador = modelo.unwrap_python_model()
    estatisticas = empacotador.motor_imobiliario.estatisticas
    metricas = MetricasServing(
        set(estatisticas.tabela_zonas), set(estatisticas.tabela_bairros)
    )
    metricas.modelo.info(
        {
            "nome": nome,
            "versao": str(versao.version),
            "alias_na_carga": alias,
            "run_id": str(versao.run_id),
        }
    )
    metricas.carregado.set_to_current_time()
    metricas.registrado.set(versao.creation_timestamp / 1000)
    run = cliente.get_run(versao.run_id)
    metricas.treinado.set((run.info.end_time or run.info.start_time) / 1000)
    return AdaptadorServing(scoring_server.init(modelo), metricas)


def executar_servidor() -> None:
    uvicorn.run(criar_aplicacao(), host="0.0.0.0", port=8080, workers=1)


if __name__ == "__main__":
    executar_servidor()
