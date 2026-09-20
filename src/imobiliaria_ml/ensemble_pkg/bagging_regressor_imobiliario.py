"""Estratégia de BaggingRegressor como ensemble homogêneo."""

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from ..modelos_pkg.regressor_protocol import RegressorProtocol
from .estrategia_ensemble import EstrategiaEnsemble


class BaggingRegressorImobiliario(EstrategiaEnsemble):
    """Ensemble homogêneo utilizando BaggingRegressor."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: float = 0.8,
        bootstrap: bool = True,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_samples = max_samples
        self._bootstrap = bootstrap

    def criar_estimador(
        self,
        estimadores_elegiveis: list[tuple[str, RegressorProtocol]],
    ) -> RegressorProtocol:
        """Cria o BaggingRegressor sobre o estimador base (ou primeiro elegível)."""
        estimator_base = (
            estimadores_elegiveis[0][1]
            if estimadores_elegiveis
            else DecisionTreeRegressor(max_depth=8, random_state=42)
        )

        return BaggingRegressor(  # type: ignore[return-value]
            estimator=estimator_base,  # type: ignore[arg-type]
            n_estimators=self._n_estimators,
            max_samples=self._max_samples,
            bootstrap=self._bootstrap,
            random_state=42,
            n_jobs=-1,
        )
