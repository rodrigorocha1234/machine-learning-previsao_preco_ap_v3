"""Registrador de modelo no MLflow Model Registry com alias @champion."""

from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from .modelo_imobiliario_pyfunc import ModeloImobiliarioPyFunc



class RegistradorModelo:
    """Gerencia o registro e a promoção de versões no Model Registry."""

    def __init__(
        self,
        nome_modelo: str = "preco-imoveis",
    ) -> None:
        self._nome_modelo = nome_modelo
        self._client = MlflowClient()

    def registrar_modelo_campeao(
        self,
        pipeline_campea: Pipeline,
        metricas: dict[str, float],
        desconto_maximo: float = 10.0,
    ) -> str:
        """Registra o modelo campeão encapsulado em PyFunc e atribui o alias @champion.

        Parameters
        ----------
        pipeline_campea : Pipeline
            Pipeline treinada completa.
        metricas : dict[str, float]
            Métricas de avaliação final.
        desconto_maximo : float
            Teto de desconto comercial permitido no serving.

        Returns
        -------
        str
            URI do modelo registrado.
        """
        modelo_pyfunc = ModeloImobiliarioPyFunc(
            modelo_pipeline=pipeline_campea,
            desconto_maximo_permitido=desconto_maximo,
        )

        run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "local"

        caminho_src = Path("src")
        code_paths = [str(caminho_src)] if caminho_src.exists() else None

        info_modelo = mlflow.pyfunc.log_model(
            artifact_path="modelo_imobiliario",
            python_model=modelo_pyfunc,
            registered_model_name=self._nome_modelo,
            code_path=code_paths,
        )

        try:
            # Obtém a versão mais recente e atribui alias champion
            versoes = self._client.search_model_versions(f"name='{self._nome_modelo}'")
            if versoes:
                versao_recente = str(versoes[0].version)
                self._client.set_registered_model_alias(
                    name=self._nome_modelo,
                    alias="champion",
                    version=versao_recente,
                )
        except Exception:
            pass

        return str(info_modelo.model_uri)
