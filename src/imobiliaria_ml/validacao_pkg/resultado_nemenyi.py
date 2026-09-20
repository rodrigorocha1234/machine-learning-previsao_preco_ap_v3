"""Estrutura do resultado do teste post-hoc de Nemenyi."""

from dataclasses import dataclass, field
import pandas as pd


@dataclass(frozen=True)
class ResultadoNemenyi:
    """Resultado do teste de comparações múltiplas de Nemenyi."""

    matriz_p_valores: pd.DataFrame = field(default_factory=pd.DataFrame)
    alpha: float = 0.05
    executado: bool = False
