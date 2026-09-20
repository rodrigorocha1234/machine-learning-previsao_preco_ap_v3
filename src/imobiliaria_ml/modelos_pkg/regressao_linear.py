"""Estratégia para Regressão Linear Simples."""

from sklearn.linear_model import LinearRegression
from ..enums_pkg.tipo_modelo import TipoModelo
from ..enums_pkg.tipo_escalonador import TipoEscalonador
from .estrategia_modelo import EstrategiaModelo
from .parametros_modelo import ParametrosModelo


class RegressaoLinear(EstrategiaModelo[LinearRegression]):
    """Estratégia para modelo de Regressão Linear."""

    def criar_modelo(self, parametros: ParametrosModelo | None = None) -> LinearRegression:
        params = parametros.valores if parametros else {}
        fit_intercept = bool(params.get("fit_intercept", True))
        return LinearRegression(fit_intercept=fit_intercept)

    def obter_grade_hiperparametros(
        self,
    ) -> dict[str, list[int | float | str | bool | tuple[int, ...] | None]]:
        return {
            "modelo__fit_intercept": [True, False],
        }

    def obter_tipo_modelo(self) -> TipoModelo:
        return TipoModelo.REGRESSAO_LINEAR

    def obter_escalonador_padrao(self) -> TipoEscalonador:
        return TipoEscalonador.STANDARD
