"""Estratégia de StackingRegressor com meta-estimador configurável."""

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from ..enums_pkg.tipo_estimador_final import TipoEstimadorFinal
from ..modelos_pkg.regressor_protocol import RegressorProtocol
from .estrategia_ensemble import EstrategiaEnsemble


class StackingRegressorImobiliario(EstrategiaEnsemble):
    """Ensemble por Stacking com meta-modelo final configurado via TipoEstimadorFinal."""

    def __init__(
        self,
        tipo_final: TipoEstimadorFinal = TipoEstimadorFinal.RIDGE,
        cv_interno: int = 5,
    ) -> None:
        self._tipo_final = tipo_final
        self._cv_interno = cv_interno

    def _obter_meta_estimador(self) -> RegressorProtocol:
        match self._tipo_final:
            case TipoEstimadorFinal.REGRESSAO_LINEAR:
                return LinearRegression()  # type: ignore[return-value]
            case TipoEstimadorFinal.RIDGE:
                return Ridge(alpha=1.0, random_state=42)  # type: ignore[return-value]
            case TipoEstimadorFinal.LASSO:
                return Lasso(alpha=0.1, random_state=42)  # type: ignore[return-value]
            case TipoEstimadorFinal.ELASTIC_NET:
                return ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)  # type: ignore[return-value]

    def criar_estimador(
        self,
        estimadores_elegiveis: list[tuple[str, RegressorProtocol]],
    ) -> RegressorProtocol:
        """Cria o StackingRegressor usando apenas os estimadores elegíveis."""
        if len(estimadores_elegiveis) < 2:
            raise ValueError("Stacking requer ao menos dois estimadores elegíveis.")

        estimadores_sklearn = [
            (nome, estimador) for nome, estimador in estimadores_elegiveis
        ]
        meta_estimador = self._obter_meta_estimador()

        return StackingRegressor(  # type: ignore[return-value]
            estimators=estimadores_sklearn,
            final_estimator=meta_estimador,  # type: ignore[arg-type]
            cv=self._cv_interno,
            n_jobs=-1,
        )
