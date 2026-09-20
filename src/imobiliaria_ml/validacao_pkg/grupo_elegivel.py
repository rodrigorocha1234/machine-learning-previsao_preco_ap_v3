"""Estrutura do grupo de modelos estatisticamente elegíveis."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GrupoElegivel:
    """Modelos que não possuem diferença estatisticamente significativa do melhor candidato."""

    melhor_modelo: str
    modelos_elegiveis: tuple[str, ...]
    rankings: dict[str, float] = field(default_factory=dict)
    rmse_medios: dict[str, float] = field(default_factory=dict)
