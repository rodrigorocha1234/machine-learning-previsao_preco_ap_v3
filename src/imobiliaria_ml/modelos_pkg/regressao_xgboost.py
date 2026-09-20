"""Estratégia para XGBoost Regressor."""

from xgboost import XGBRegressor
from ..enums_pkg.tipo_modelo import TipoModelo
from ..enums_pkg.tipo_escalonador import TipoEscalonador
from .estrategia_modelo import EstrategiaModelo
from .parametros_modelo import ParametrosModelo


class RegressaoXgboost(EstrategiaModelo[XGBRegressor]):
    """Estratégia para XGBRegressor."""

    def criar_modelo(self, parametros: ParametrosModelo | None = None) -> XGBRegressor:
        params = parametros.valores if parametros else {}
        n_estimators = int(params.get("n_estimators", 100))
        max_depth = int(params.get("max_depth", 5))
        learning_rate = float(params.get("learning_rate", 0.1))
        subsample = float(params.get("subsample", 1.0))
        colsample = float(params.get("colsample_bytree", 1.0))
        return XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample,
            random_state=42,
            n_jobs=-1,
        )

    def obter_grade_hiperparametros(
        self,
    ) -> dict[str, list[int | float | str | bool | tuple[int, ...] | None]]:
        return {
            "modelo__n_estimators": [200, 500],
            "modelo__max_depth": [3, 5, 8],
            "modelo__learning_rate": [0.01, 0.05, 0.1],
            "modelo__subsample": [0.7, 0.9, 1.0],
            "modelo__colsample_bytree": [0.7, 0.9, 1.0],
        }

    def obter_tipo_modelo(self) -> TipoModelo:
        return TipoModelo.XGBOOST

    def obter_escalonador_padrao(self) -> TipoEscalonador:
        return TipoEscalonador.SEM_ESCALA
