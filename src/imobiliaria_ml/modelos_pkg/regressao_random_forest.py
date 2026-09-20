"""Estratégia para Random Forest Regressor."""

from sklearn.ensemble import RandomForestRegressor
from ..enums_pkg.tipo_modelo import TipoModelo
from ..enums_pkg.tipo_escalonador import TipoEscalonador
from .estrategia_modelo import EstrategiaModelo
from .parametros_modelo import ParametrosModelo


class RegressaoRandomForest(EstrategiaModelo[RandomForestRegressor]):
    """Estratégia para modelo RandomForestRegressor."""

    def criar_modelo(self, parametros: ParametrosModelo | None = None) -> RandomForestRegressor:
        params = parametros.valores if parametros else {}
        n_estimators = int(params.get("n_estimators", 100))
        max_depth = int(params["max_depth"]) if params.get("max_depth") is not None else None
        min_samples_leaf = int(params.get("min_samples_leaf", 1))
        max_features = params.get("max_features", "sqrt")
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1,
        )

    def obter_grade_hiperparametros(
        self,
    ) -> dict[str, list[int | float | str | bool | tuple[int, ...] | None]]:
        return {
            "modelo__n_estimators": [200, 500],
            "modelo__max_depth": [8, 12, 20, None],
            "modelo__min_samples_leaf": [1, 2, 5],
            "modelo__max_features": ["sqrt", 0.7, 1.0],
        }

    def obter_tipo_modelo(self) -> TipoModelo:
        return TipoModelo.RANDOM_FOREST

    def obter_escalonador_padrao(self) -> TipoEscalonador:
        return TipoEscalonador.SEM_ESCALA
