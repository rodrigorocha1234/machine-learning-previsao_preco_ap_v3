"""Estrutura para armazenar as estatísticas descritivas completas da EDA."""

from dataclasses import dataclass, field
import pandas as pd


@dataclass(frozen=True)
class EstatisticasDescritivas:
    """Estatísticas descritivas completas calculadas sobre o dataset."""

    tabela_numerica: dict[str, dict[str, float | int]] = field(default_factory=dict)
    tabela_categorica: dict[str, dict[str, float | int | str]] = field(default_factory=dict)
    tabela_vif: dict[str, float] = field(default_factory=dict)
    outliers_iqr: dict[str, int] = field(default_factory=dict)
    analise_zona: dict[str, dict[str, float | int | str]] = field(default_factory=dict)
    diagnostico_zona: dict[str, str] = field(default_factory=dict)
    tabela_zonas_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    resumo_zonas_md: str = ""
    relatorio_negocio_md: str = ""


