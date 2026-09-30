from dataclasses import dataclass
from typing import ClassVar, Final

import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class ResultadoAnova:
    estatistica_f: float
    p_valor: float
    significativo: bool


class TesteTukey:
    __test__: ClassVar[bool] = False

    def __init__(self, limiar_alfa: float = 0.05) -> None:
        self._alfa: Final[float] = limiar_alfa

    def testar_anova(self, matriz_scores: pd.DataFrame) -> ResultadoAnova:
        colunas = list(matriz_scores.columns)
        amostras = [matriz_scores[col].to_numpy() for col in colunas]
        resultado = stats.f_oneway(*amostras)

        f_stat = float(resultado.statistic)
        p_val = float(resultado.pvalue)
        eh_sig = bool(p_val < self._alfa)

        return ResultadoAnova(
            estatistica_f=f_stat,
            p_valor=p_val,
            significativo=eh_sig,
        )
