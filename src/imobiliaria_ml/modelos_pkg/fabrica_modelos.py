"""Fábrica para instanciar estratégias de modelos por TipoModelo."""

from ..enums_pkg.tipo_modelo import TipoModelo
from .regressor_protocol import RegressorProtocol
from .estrategia_modelo import EstrategiaModelo
from .regressao_linear import RegressaoLinear
from .regressao_multipla import RegressaoMultipla
from .regressao_polinomial import RegressaoPolinomial
from .regressao_ridge import RegressaoRidge
from .regressao_lasso import RegressaoLasso
from .regressao_elastic_net import RegressaoElasticNet
from .regressao_arvore import RegressaoArvore
from .regressao_random_forest import RegressaoRandomForest
from .regressao_gradient_boosting import RegressaoGradientBoosting
from .regressao_svr import RegressaoSvr
from .regressao_rede_neural import RegressaoRedeNeural
from .regressao_xgboost import RegressaoXgboost
from .regressao_lightgbm import RegressaoLightgbm
from .regressao_catboost import RegressaoCatboost


class FabricaModelos:
    """Fábrica de estratégias de algoritmos de regressão."""

    def criar(self, tipo: TipoModelo) -> EstrategiaModelo[RegressorProtocol]:
        """Instancia a estratégia de modelo apropriada para o TipoModelo informado.

        Parameters
        ----------
        tipo : TipoModelo
            Algoritmo selecionado.

        Returns
        -------
        EstrategiaModelo[RegressorProtocol]
            Estratégia correspondente.
        """
        match tipo:
            case TipoModelo.REGRESSAO_LINEAR:
                return RegressaoLinear()  # type: ignore[return-value]
            case TipoModelo.REGRESSAO_MULTIPLA:
                return RegressaoMultipla()  # type: ignore[return-value]
            case TipoModelo.REGRESSAO_POLINOMIAL:
                return RegressaoPolinomial()  # type: ignore[return-value]
            case TipoModelo.RIDGE:
                return RegressaoRidge()  # type: ignore[return-value]
            case TipoModelo.LASSO:
                return RegressaoLasso()  # type: ignore[return-value]
            case TipoModelo.ELASTIC_NET:
                return RegressaoElasticNet()  # type: ignore[return-value]
            case TipoModelo.ARVORE_DECISAO:
                return RegressaoArvore()  # type: ignore[return-value]
            case TipoModelo.RANDOM_FOREST:
                return RegressaoRandomForest()  # type: ignore[return-value]
            case TipoModelo.GRADIENT_BOOSTING:
                return RegressaoGradientBoosting()  # type: ignore[return-value]
            case TipoModelo.SVR:
                return RegressaoSvr()  # type: ignore[return-value]
            case TipoModelo.REDE_NEURAL:
                return RegressaoRedeNeural()  # type: ignore[return-value]
            case TipoModelo.XGBOOST:
                return RegressaoXgboost()  # type: ignore[return-value]
            case TipoModelo.LIGHTGBM:
                return RegressaoLightgbm()  # type: ignore[return-value]
            case TipoModelo.CATBOOST:
                return RegressaoCatboost()  # type: ignore[return-value]
