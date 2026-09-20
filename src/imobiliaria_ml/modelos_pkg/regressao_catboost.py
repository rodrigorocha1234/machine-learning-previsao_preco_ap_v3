"""Estratégia para CatBoost Regressor."""

from catboost import CatBoostRegressor
from ..enums_pkg.tipo_modelo import TipoModelo
from ..enums_pkg.tipo_escalonador import TipoEscalonador
from .estrategia_modelo import EstrategiaModelo
from .parametros_modelo import ParametrosModelo


class RegressaoCatboost(EstrategiaModelo[CatBoostRegressor]):
    """Estratégia para CatBoostRegressor."""

    def criar_modelo(self, parametros: ParametrosModelo | None = None) -> CatBoostRegressor:
        params = parametros.valores if parametros else {}
        iterations = int(params.get("iterations", 200))
        depth = int(params.get("depth", 6))
        learning_rate = float(params.get("learning_rate", 0.1))
        l2_reg = float(params.get("l2_leaf_reg", 3.0))
        return CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_reg,
            random_seed=42,
            verbose=False,
        )

    def obter_grade_hiperparametros(
        self,
    ) -> dict[str, list[int | float | str | bool | tuple[int, ...] | None]]:
        return {
            "modelo__iterations": [200, 500],
            "modelo__depth": [4, 6, 8],
            "modelo__learning_rate": [0.01, 0.05, 0.1],
            "modelo__l2_leaf_reg": [1.0, 3.0, 5.0, 10.0],
            "modelo__verbose": [False],
        }

    def obter_tipo_modelo(self) -> TipoModelo:
        return TipoModelo.CATBOOST

    def obter_escalonador_padrao(self) -> TipoEscalonador:
        return TipoEscalonador.SEM_ESCALA
