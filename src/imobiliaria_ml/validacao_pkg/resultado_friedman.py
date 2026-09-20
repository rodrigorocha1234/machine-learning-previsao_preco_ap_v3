"""Estrutura do resultado do teste estatístico de Friedman."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResultadoFriedman:
    """Resultado do teste não-paramétrico de Friedman."""

    estatistica: float
    p_valor: float
    alpha: float
    significativo: bool
    rankings_medios: dict[str, float] = field(default_factory=dict)
