"""Estrutura representativa de um modelo candidato no pipeline."""

from dataclasses import dataclass, field
from sklearn.pipeline import Pipeline
from ..enums_pkg.tipo_modelo import TipoModelo


@dataclass(frozen=True)
class CandidatoModelo:
    """Encapsula o pipeline e os metadados de um modelo candidato validado."""

    tipo_modelo: TipoModelo
    nome: str
    pipeline: Pipeline
    melhores_parametros: dict[str, int | float | str | bool | tuple[int, ...] | None] = field(
        default_factory=dict
    )
    rmse_cv: float = 0.0
