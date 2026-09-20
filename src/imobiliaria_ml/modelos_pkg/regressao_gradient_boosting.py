"""Estratégia para Gradient Boosting Regressor."""

from sklearn.ensemble import GradientBoostingRegressor
from ..enums_pkg.tipo_modelo import TipoModelo
from ..enums_pkg.tipo_escalonador import TipoEscalonador
from .estrategia_modelo import EstrategiaModelo
from .parametros_modelo import ParametrosModelo


class RegressaoGradientBoosting(EstrategiaModelo[GradientBoostingRegressor]):
    """Estratégia para GradientBoostingRegressor."""

    def criar_modelo(self, parametros: ParametrosModelo | None = None) -> GradientBoostingRegressor:
        params = parametros.valores if parametros else {}
        n_estimators = int(params.get("n_estimators", 100))
        learning_rate = float(params.get("learning_rate", 0.1))
        max_depth = int(params.get("max_depth", 3))
        min_samples_leaf = int(params.get("min_samples_leaf", 1))
        subsample = float(params.get("subsample", 1.0))
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=42,
        )

    def obter_grade_hiperparametros(
        self,
    ) -> dict[str, list[int | float | str | bool | tuple[int, ...] | None]]:
        return {
            "modelo__n_estimators": [100, 200, 500],
            "modelo__learning_rate": [0.01, 0.05, 0.1],
            "modelo__max_depth": [2, 3, 5],
            "modelo__min_samples_leaf": [1, 2, 5],
            "modelo__subsample": [0.7, 0.9, 1.0],
        }

    def obter_tipo_modelo(self) -> TipoModelo:
        return TipoModelo.GRADIENT_BOOSTING

    def obter_escalonador_padrao(self) -> TipoEscalonador:
        return TipoEscalonador.SEM_ESCALA
