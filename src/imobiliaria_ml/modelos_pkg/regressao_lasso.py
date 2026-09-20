"""Estratégia para Regressão Lasso."""

from sklearn.linear_model import Lasso
from ..enums_pkg.tipo_modelo import TipoModelo
from ..enums_pkg.tipo_escalonador import TipoEscalonador
from .estrategia_modelo import EstrategiaModelo
from .parametros_modelo import ParametrosModelo


class RegressaoLasso(EstrategiaModelo[Lasso]):
    """Estratégia para regressão com regularização L1 (Lasso)."""

    def criar_modelo(self, parametros: ParametrosModelo | None = None) -> Lasso:
        params = parametros.valores if parametros else {}
        alpha = float(params.get("alpha", 1.0))
        max_iter = int(params.get("max_iter", 10000))
        tol = float(params.get("tol", 0.001))
        return Lasso(alpha=alpha, max_iter=max_iter, tol=tol, random_state=42)

    def obter_grade_hiperparametros(
        self,
    ) -> dict[str, list[int | float | str | bool | tuple[int, ...] | None]]:
        return {
            "modelo__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
        }

    def obter_tipo_modelo(self) -> TipoModelo:
        return TipoModelo.LASSO

    def obter_escalonador_padrao(self) -> TipoEscalonador:
        return TipoEscalonador.STANDARD
