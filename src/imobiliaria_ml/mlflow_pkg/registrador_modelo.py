import os
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from ..negocio_pkg.metricas_negocio_resultado import MetricasNegocioResultado
from .modelo_imobiliario_pyfunc import ModeloImobiliarioPyFunc


class RegistradorModelo:
    """Gerencia o registro e a promoção de versões no Model Registry."""

    def __init__(
        self,
        nome_modelo: str = "preco-imoveis",
        tracking_uri: str | None = None,
        usar_votacao: bool = False,
    ) -> None:
        self._nome_modelo = nome_modelo
        self._tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self._client = MlflowClient(tracking_uri=self._tracking_uri)
        self._usar_votacao = usar_votacao

    def registrar_modelo_campeao(
        self,
        pipeline_campea: Pipeline,
        metricas: dict[str, float],
        desconto_maximo: float = 10.0,
        nome_modelo_concreto: str = "",
        metricas_negocio: MetricasNegocioResultado | None = None,
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
        nome_modelo_concreto : str
            Nome do algoritmo / estimador campeão (ex: ridge, xgboost, ensemble_voting).
        metricas_negocio : MetricasNegocioResultado | None
            Métricas de negócio financeiro e operacional da imobiliária.

        Returns
        -------
        str
            URI do modelo registrado.
        """
        modelo_pyfunc = ModeloImobiliarioPyFunc(
            modelo_pipeline=pipeline_campea,
            desconto_maximo_permitido=desconto_maximo,
        )

        if mlflow.active_run():
            sufixo = "_votacao" if self._usar_votacao else ""
            nome_run = (f"campeao_{nome_modelo_concreto}" if nome_modelo_concreto else "campeao") + sufixo
            tags_registro: dict[str, str] = {
                "modelo": nome_modelo_concreto or "campeao",
                "mlflow.runName": nome_run,
                "status_registro": "registrado",
                "alias_registro": "champion",
                "tipo_empacotamento": "pyfunc",
                "usar_votacao": str(self._usar_votacao).lower(),
            }
            if metricas_negocio is not None:
                tags_registro["desconto_seguro"] = f"{metricas_negocio.desconto_seguro_recomendado:.2f}%"
                tags_registro["cobertura_10"] = f"{metricas_negocio.cobertura_10 * 100:.2f}%"
            mlflow.set_tags(tags_registro)

        run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "local"

        info_modelo = mlflow.pyfunc.log_model(
            artifact_path="modelo_imobiliario",
            python_model=modelo_pyfunc,
            registered_model_name=self._nome_modelo,
        )

        try:
            versao_recente: str | None = None
            if hasattr(info_modelo, "registered_model_version") and info_modelo.registered_model_version:
                versao_recente = str(info_modelo.registered_model_version)
            else:
                versoes = self._client.search_model_versions(f"name='{self._nome_modelo}'")
                if versoes:
                    versao_recente = str(versoes[0].version)

            if versao_recente:
                self._client.set_registered_model_alias(
                    name=self._nome_modelo,
                    alias="champion",
                    version=versao_recente,
                )
                tags_versao: dict[str, str] = {
                    "modelo": nome_modelo_concreto or "campeao",
                    "desconto_maximo": str(desconto_maximo),
                    "alvo": "Valor_da_Venda",
                    "framework": "scikit-learn",
                    "status": "champion",
                    "usar_votacao": str(self._usar_votacao).lower(),
                }
                if metricas_negocio is not None:
                    tags_versao["negocio_desconto_seguro"] = f"{metricas_negocio.desconto_seguro_recomendado:.2f}%"
                    tags_versao["negocio_cobertura_5"] = f"{metricas_negocio.cobertura_5 * 100:.2f}%"
                    tags_versao["negocio_cobertura_10"] = f"{metricas_negocio.cobertura_10 * 100:.2f}%"
                    tags_versao["negocio_cobertura_15"] = f"{metricas_negocio.cobertura_15 * 100:.2f}%"
                    tags_versao["negocio_mae_reais"] = f"R$ {metricas_negocio.mae_reais:,.2f}"
                    tags_versao["negocio_vies_medio"] = f"R$ {metricas_negocio.vies_medio:,.2f}"
                    tags_versao["negocio_risco_subprecificacao"] = f"{metricas_negocio.risco_subprecificacao * 100:.2f}%"
                    tags_versao["negocio_risco_superprecificacao"] = f"{metricas_negocio.risco_superprecificacao * 100:.2f}%"
                    tags_versao["negocio_margem_negociacao"] = f"R$ {metricas_negocio.margem_negociacao_estimada:,.2f}"
                    tags_versao["negocio_receita_potencial_perdida"] = f"R$ {metricas_negocio.receita_potencial_perdida:,.2f}"
                for tk, tv in tags_versao.items():
                    try:
                        self._client.set_model_version_tag(
                            name=self._nome_modelo,
                            version=versao_recente,
                            key=tk,
                            value=tv,
                        )
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            mlflow.end_run()
        except Exception:
            pass

        return str(getattr(info_modelo, "model_uri", f"models:/{self._nome_modelo}@champion"))
