"""Fábrica de estratégias de ensemble a partir de TipoEnsemble."""

from ..enums_pkg.tipo_ensemble import TipoEnsemble
from ..enums_pkg.tipo_estimador_final import TipoEstimadorFinal
from .estrategia_ensemble import EstrategiaEnsemble
from .votacao_regressor import VotacaoRegressor
from .votacao_regressor_ponderada import VotacaoRegressorPonderada
from .stacking_regressor_imobiliario import StackingRegressorImobiliario
from .bagging_regressor_imobiliario import BaggingRegressorImobiliario


class FabricaEnsemble:
    """Factory para instanciação de estratégias de ensemble."""

    def criar(
        self,
        tipo: TipoEnsemble,
        pesos_por_modelo: dict[str, float] | None = None,
        estimador_final: TipoEstimadorFinal = TipoEstimadorFinal.RIDGE,
    ) -> EstrategiaEnsemble:
        """Instancia a estratégia de ensemble solicitada.

        Parameters
        ----------
        tipo : TipoEnsemble
            Técnica de ensemble desejada.
        pesos_por_modelo : dict[str, float] | None
            Pesos por modelo para estratégias ponderadas.
        estimador_final : TipoEstimadorFinal
            Meta-estimador para Stacking.

        Returns
        -------
        EstrategiaEnsemble
            Estratégia configurada.
        """
        match tipo:
            case TipoEnsemble.VOTING_MEDIA:
                return VotacaoRegressor()
            case TipoEnsemble.VOTING_PONDERADO_RMSE | TipoEnsemble.VOTING_PONDERADO_RANKING:
                return VotacaoRegressorPonderada(pesos_por_modelo or {})
            case TipoEnsemble.STACKING:
                return StackingRegressorImobiliario(tipo_final=estimador_final)
            case TipoEnsemble.BAGGING:
                return BaggingRegressorImobiliario()
