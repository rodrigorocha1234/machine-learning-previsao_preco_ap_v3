"""Estratégia para Regressão Elastic Net."""

from sklearn.linear_model import ElasticNet
from ..enums_pkg.tipo_modelo import TipoModelo
from ..enums_pkg.tipo_escalonador import TipoEscalonador
from .estrategia_modelo import EstrategiaModelo
from .parametros_modelo import ParametrosModelo


class RegressaoElasticNet(EstrategiaModelo[ElasticNet]):
    """Estratégia para regressão regularizada Elastic Net (L1 + L2)."""

    def criar_modelo(self, parametros: ParametrosModelo | None = None) -> ElasticNet:
        params = parametros.valores if parametros else {}
        alpha = float(params.get("alpha", 1.0))
        l1_ratio = float(params.get("l1_ratio", 0.5))
        max_iter = int(params.get("max_iter", 10000))
        tol = float(params.get("tol", 0.001))
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol, random_state=42)

    def obter_grade_hiperparametros(
        self,
    ) -> dict[str, list[int | float | str | bool | tuple[int, ...] | None]]:
        return {
            "modelo__alpha": [0.001, 0.01, 0.1, 1.0],
            "modelo__l1_ratio": [0.1, 0.25, 0.5, 0.75, 0.9],
        }

    def obter_tipo_modelo(self) -> TipoModelo:
        return TipoModelo.ELASTIC_NET

    def obter_escalonador_padrao(self) -> TipoEscalonador:
        return TipoEscalonador.STANDARD
