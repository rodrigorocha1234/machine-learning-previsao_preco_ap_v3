"""Ponto de entrada para validação de gates e promoção de modelos no Model Registry."""

from mlflow.tracking import MlflowClient
from ..enums_pkg.nivel_drift import NivelDrift
from ..drift_pkg.resultado_drift import ResultadoDrift


class PromoverModelo:
    """Aplica os critérios de aceite e gates de qualidade para promover versão a @champion."""

    def __init__(self, nome_modelo: str = "preco-imoveis") -> None:
        self._nome_modelo = nome_modelo
        self._client = MlflowClient()

    def avaliar_promocao(
        self,
        versao_candidata: str,
        resultado_drift: ResultadoDrift,
        rmse_candidato: float,
        rmse_campeao_atual: float,
    ) -> bool:
        """Avalia os gates de governança e promove o modelo se aprovado.

        Parameters
        ----------
        versao_candidata : str
            Versão do modelo recém-treinado no Registry.
        resultado_drift : ResultadoDrift
            Diagnóstico de drift na data de avaliação.
        rmse_candidato : float
            RMSE no holdout do modelo candidato.
        rmse_campeao_atual : float
            RMSE do modelo atualmente promovido.

        Returns
        -------
        bool
            True se o modelo foi aprovado e promovido para @champion.
        """
        # Gate 1: Rejeitar se houver drift forte não controlado
        if resultado_drift.nivel_drift == NivelDrift.FORTE:
            return False

        # Gate 2: Não permitir regressão material de desempenho (> 2% pior em RMSE)
        if rmse_candidato > (rmse_campeao_atual * 1.02):
            return False

        # Gate 3: Promoção oficial no MLflow Model Registry
        try:
            self._client.set_registered_model_alias(
                name=self._nome_modelo,
                alias="champion",
                version=versao_candidata,
            )
            return True
        except Exception:
            return False


if __name__ == "__main__":
    app = PromoverModelo()
