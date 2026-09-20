"""Estrutura do resultado consolidado do monitoramento de Data Drift."""

from dataclasses import dataclass, field
from ..enums_pkg.nivel_drift import NivelDrift


@dataclass(frozen=True)
class ResultadoDrift:
    """Diagnóstico de estabilidade e detecção de desvio de dados."""

    nivel_drift: NivelDrift
    drift_detectado: bool
    estatisticas_numericas: dict[str, dict[str, float]] = field(default_factory=dict)
    estatisticas_categoricas: dict[str, dict[str, float | int | str]] = field(
        default_factory=dict
    )
    drift_predicao_psi: float = 0.0
    detalhes: dict[str, str | float | int] = field(default_factory=dict)
