from typing import ClassVar, Final

import pandas as pd
from scipy import stats

from app_build.estatistica_modelos.contrato_estatistica import ResultadoFriedman


class TesteFriedman:
    __test__: ClassVar[bool] = False

    def __init__(self, limiar_alfa: float = 0.05) -> None:
        self._alfa: Final[float] = limiar_alfa

    def testar(self, matriz_scores: pd.DataFrame) -> ResultadoFriedman:
        colunas = list(matriz_scores.columns)
        amostras = [matriz_scores[col].to_numpy() for col in colunas]
        resultado = stats.friedmanchisquare(*amostras)

        ranks_por_linha = matriz_scores.rank(axis=1, ascending=True)
        ranks_medios = {col: float(ranks_por_linha[col].mean()) for col in colunas}

        estatistica = float(resultado.statistic)
        p_valor = float(resultado.pvalue)
        eh_significativo = bool(p_valor < self._alfa)

        return ResultadoFriedman(
            estatistica=estatistica,
            p_valor=p_valor,
            eh_significativo=eh_significativo,
            ranks_medios=ranks_medios,
        )
