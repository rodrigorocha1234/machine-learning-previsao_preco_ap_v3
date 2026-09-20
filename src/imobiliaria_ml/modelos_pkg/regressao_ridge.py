"""Estratégia para Regressão Ridge."""

from sklearn.linear_model import Ridge
from ..enums_pkg.tipo_modelo import TipoModelo
from ..enums_pkg.tipo_escalonador import TipoEscalonador
from .estrategia_modelo import EstrategiaModelo
from .parametros_modelo import ParametrosModelo


class RegressaoRidge(EstrategiaModelo[Ridge]):
    """Estratégia para regressão com regularização L2 (Ridge)."""

    def criar_modelo(self, parametros: ParametrosModelo | None = None) -> Ridge:
        params = parametros.valores if parametros else {}
        alpha = float(params.get("alpha", 1.0))
        return Ridge(alpha=alpha, random_state=42)

    def obter_grade_hiperparametros(
        self,
    ) -> dict[str, list[int | float | str | bool | tuple[int, ...] | None]]:
        return {
            "modelo__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        }

    def obter_tipo_modelo(self) -> TipoModelo:
        return TipoModelo.RIDGE

    def obter_escalonador_padrao(self) -> TipoEscalonador:
        return TipoEscalonador.STANDARD
