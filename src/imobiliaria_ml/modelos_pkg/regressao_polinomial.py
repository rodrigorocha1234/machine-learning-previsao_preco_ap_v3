"""Estratégia para Regressão Polinomial."""

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from ..enums_pkg.tipo_modelo import TipoModelo
from ..enums_pkg.tipo_escalonador import TipoEscalonador
from .estrategia_modelo import EstrategiaModelo
from .parametros_modelo import ParametrosModelo


class RegressaoPolinomial(EstrategiaModelo[Pipeline]):
    """Estratégia para Regressão Polinomial composta por PolynomialFeatures + LinearRegression."""

    def criar_modelo(self, parametros: ParametrosModelo | None = None) -> Pipeline:
        params = parametros.valores if parametros else {}
        grau = int(params.get("degree", 2))
        fit_intercept = bool(params.get("fit_intercept", True))
        return Pipeline(
            steps=[
                ("poly", PolynomialFeatures(degree=grau, include_bias=False)),
                ("linear", LinearRegression(fit_intercept=fit_intercept)),
            ]
        )

    def obter_grade_hiperparametros(
        self,
    ) -> dict[str, list[int | float | str | bool | tuple[int, ...] | None]]:
        return {
            "modelo__poly__degree": [2, 3],
            "modelo__linear__fit_intercept": [True, False],
        }

    def obter_tipo_modelo(self) -> TipoModelo:
        return TipoModelo.REGRESSAO_POLINOMIAL

    def obter_escalonador_padrao(self) -> TipoEscalonador:
        return TipoEscalonador.STANDARD
