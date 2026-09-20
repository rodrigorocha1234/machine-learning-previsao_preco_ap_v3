"""Estratégia para Árvore de Decisão."""

from sklearn.tree import DecisionTreeRegressor
from ..enums_pkg.tipo_modelo import TipoModelo
from ..enums_pkg.tipo_escalonador import TipoEscalonador
from .estrategia_modelo import EstrategiaModelo
from .parametros_modelo import ParametrosModelo


class RegressaoArvore(EstrategiaModelo[DecisionTreeRegressor]):
    """Estratégia para regressão baseada em Árvore de Decisão."""

    def criar_modelo(self, parametros: ParametrosModelo | None = None) -> DecisionTreeRegressor:
        params = parametros.valores if parametros else {}
        max_depth = int(params["max_depth"]) if params.get("max_depth") is not None else None
        min_samples_split = int(params.get("min_samples_split", 2))
        min_samples_leaf = int(params.get("min_samples_leaf", 1))
        return DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )

    def obter_grade_hiperparametros(
        self,
    ) -> dict[str, list[int | float | str | bool | tuple[int, ...] | None]]:
        return {
            "modelo__max_depth": [3, 5, 8, 12, None],
            "modelo__min_samples_split": [2, 5, 10, 20],
            "modelo__min_samples_leaf": [1, 2, 5, 10],
        }

    def obter_tipo_modelo(self) -> TipoModelo:
        return TipoModelo.ARVORE_DECISAO

    def obter_escalonador_padrao(self) -> TipoEscalonador:
        return TipoEscalonador.SEM_ESCALA
