"""Estrutura para armazenar as estatísticas descritivas completas da EDA."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EstatisticasDescritivas:
    """Estatísticas descritivas completas calculadas sobre o dataset."""

    tabela_numerica: dict[str, dict[str, float | int]] = field(default_factory=dict)
    tabela_categorica: dict[str, dict[str, float | int | str]] = field(default_factory=dict)
    tabela_vif: dict[str, float] = field(default_factory=dict)
    outliers_iqr: dict[str, int] = field(default_factory=dict)
