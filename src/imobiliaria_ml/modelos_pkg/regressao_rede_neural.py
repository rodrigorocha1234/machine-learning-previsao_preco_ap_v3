"""Estratégia para Rede Neural Artificial (MLPRegressor)."""

from sklearn.neural_network import MLPRegressor
from ..enums_pkg.tipo_modelo import TipoModelo
from ..enums_pkg.tipo_escalonador import TipoEscalonador
from .estrategia_modelo import EstrategiaModelo
from .parametros_modelo import ParametrosModelo


class RegressaoRedeNeural(EstrategiaModelo[MLPRegressor]):
    """Estratégia para MLPRegressor."""

    def criar_modelo(self, parametros: ParametrosModelo | None = None) -> MLPRegressor:
        params = parametros.valores if parametros else {}
        hidden = params.get("hidden_layer_sizes", (64, 32))
        hidden_tuple = tuple(hidden) if isinstance(hidden, (list, tuple)) else (64, 32)
        alpha = float(params.get("alpha", 0.001))
        lr_init = float(params.get("learning_rate_init", 0.001))
        max_iter = int(params.get("max_iter", 500))
        early_stopping = bool(params.get("early_stopping", False))
        return MLPRegressor(
            hidden_layer_sizes=hidden_tuple,
            alpha=alpha,
            learning_rate_init=lr_init,
            max_iter=max_iter,
            random_state=42,
            early_stopping=early_stopping,
        )

    def obter_grade_hiperparametros(
        self,
    ) -> dict[str, list[int | float | str | bool | tuple[int, ...] | None]]:
        return {
            "modelo__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
            "modelo__alpha": [0.0001, 0.001, 0.01],
            "modelo__learning_rate_init": [0.0005, 0.001, 0.01],
        }

    def obter_tipo_modelo(self) -> TipoModelo:
        return TipoModelo.REDE_NEURAL

    def obter_escalonador_padrao(self) -> TipoEscalonador:
        return TipoEscalonador.STANDARD
