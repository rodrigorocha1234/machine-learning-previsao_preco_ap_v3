"""Resultado estruturado da validação cruzada RepeatedKFold."""

from dataclasses import dataclass, field
import pandas as pd


@dataclass(frozen=True)
class ResultadoValidacaoCruzada:
    """Consolidação dos 150 folds e da matriz de 30 repetições por modelo."""

    tabela_folds: pd.DataFrame = field(default_factory=pd.DataFrame)
    matriz_repeticoes: pd.DataFrame = field(default_factory=pd.DataFrame)
    resumo_modelos: pd.DataFrame = field(default_factory=pd.DataFrame)
