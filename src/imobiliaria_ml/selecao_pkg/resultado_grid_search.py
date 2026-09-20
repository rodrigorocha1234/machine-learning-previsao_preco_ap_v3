"""Resultado estruturado da execução do GridSearchCV."""

from dataclasses import dataclass, field
import pandas as pd
from ..enums_pkg.tipo_modelo import TipoModelo


@dataclass(frozen=True)
class ResultadoGridSearch:
    """Encapsula as conclusões e artefatos de um processo de GridSearch."""

    tipo_modelo: TipoModelo
    melhores_parametros: dict[str, int | float | str | bool | tuple[int, ...] | None]
    melhor_score_rmse: float
    tabela_cv_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    tempo_total_segundos: float = 0.0
