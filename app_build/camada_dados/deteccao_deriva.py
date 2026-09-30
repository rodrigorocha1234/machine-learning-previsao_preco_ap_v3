from dataclasses import dataclass
from typing import Final

import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class ResultadoDerivaColuna:
    coluna: str
    estatistica: float
    p_valor: float
    tem_deriva: bool


@dataclass(frozen=True)
class RelatorioDeriva:
    resultados: tuple[ResultadoDerivaColuna, ...]
    percentual_colunas_com_deriva: float


class DeteccaoDeriva:
    def __init__(self, limiar_significancia: float = 0.05) -> None:
        self._limiar: Final[float] = limiar_significancia

    def comparar(
        self,
        base_referencia: pd.DataFrame,
        base_atual: pd.DataFrame,
        colunas: tuple[str, ...],
    ) -> RelatorioDeriva:
        def testar_coluna(coluna: str) -> ResultadoDerivaColuna:
            amostra_ref = base_referencia[coluna].dropna().to_numpy()
            amostra_atual = base_atual[coluna].dropna().to_numpy()
            resultado_ks = stats.ks_2samp(amostra_ref, amostra_atual)
            p_valor = float(resultado_ks.pvalue)
            estatistica = float(resultado_ks.statistic)
            tem_deriva = bool(p_valor < self._limiar)
            return ResultadoDerivaColuna(
                coluna=coluna,
                estatistica=estatistica,
                p_valor=p_valor,
                tem_deriva=tem_deriva,
            )

        resultados = tuple(map(testar_coluna, colunas))
        total_com_deriva = sum(int(r.tem_deriva) for r in resultados)
        percentual = (
            float(total_com_deriva / len(resultados)) if len(resultados) > 0 else 0.0
        )

        return RelatorioDeriva(
            resultados=resultados,
            percentual_colunas_com_deriva=percentual,
        )
