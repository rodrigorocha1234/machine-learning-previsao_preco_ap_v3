"""Estratégia para Support Vector Regression (SVR)."""

from sklearn.svm import SVR
from ..enums_pkg.tipo_modelo import TipoModelo
from ..enums_pkg.tipo_escalonador import TipoEscalonador
from .estrategia_modelo import EstrategiaModelo
from .parametros_modelo import ParametrosModelo


class RegressaoSvr(EstrategiaModelo[SVR]):
    """Estratégia para regressão com SVR."""

    def criar_modelo(self, parametros: ParametrosModelo | None = None) -> SVR:
        params = parametros.valores if parametros else {}
        kernel = str(params.get("kernel", "rbf"))
        c_val = float(params.get("C", 1.0))
        epsilon = float(params.get("epsilon", 0.1))
        gamma = str(params.get("gamma", "scale"))
        return SVR(kernel=kernel, C=c_val, epsilon=epsilon, gamma=gamma)

    def obter_grade_hiperparametros(
        self,
    ) -> dict[str, list[int | float | str | bool | tuple[int, ...] | None]]:
        return {
            "modelo__kernel": ["rbf", "linear"],
            "modelo__C": [1.0, 10.0, 100.0],
            "modelo__epsilon": [0.01, 0.1, 0.5],
            "modelo__gamma": ["scale", "auto"],
        }

    def obter_tipo_modelo(self) -> TipoModelo:
        return TipoModelo.SVR

    def obter_escalonador_padrao(self) -> TipoEscalonador:
        return TipoEscalonador.STANDARD
