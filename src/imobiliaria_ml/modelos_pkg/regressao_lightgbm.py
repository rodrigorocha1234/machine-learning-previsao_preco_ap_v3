"""Estratégia para LightGBM Regressor."""

from lightgbm import LGBMRegressor
from ..enums_pkg.tipo_modelo import TipoModelo
from ..enums_pkg.tipo_escalonador import TipoEscalonador
from .estrategia_modelo import EstrategiaModelo
from .parametros_modelo import ParametrosModelo


class RegressaoLightgbm(EstrategiaModelo[LGBMRegressor]):
    """Estratégia para LGBMRegressor."""

    def criar_modelo(self, parametros: ParametrosModelo | None = None) -> LGBMRegressor:
        params = parametros.valores if parametros else {}
        n_estimators = int(params.get("n_estimators", 100))
        num_leaves = int(params.get("num_leaves", 31))
        learning_rate = float(params.get("learning_rate", 0.1))
        min_child_samples = int(params.get("min_child_samples", 20))
        subsample = float(params.get("subsample", 1.0))
        return LGBMRegressor(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            subsample=subsample,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )

    def obter_grade_hiperparametros(
        self,
    ) -> dict[str, list[int | float | str | bool | tuple[int, ...] | None]]:
        return {
            "modelo__n_estimators": [200, 500],
            "modelo__num_leaves": [15, 31, 63],
            "modelo__learning_rate": [0.01, 0.05, 0.1],
            "modelo__min_child_samples": [10, 20, 40],
            "modelo__subsample": [0.7, 0.9, 1.0],
        }

    def obter_tipo_modelo(self) -> TipoModelo:
        return TipoModelo.LIGHTGBM

    def obter_escalonador_padrao(self) -> TipoEscalonador:
        return TipoEscalonador.SEM_ESCALA
