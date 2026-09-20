"""Execução do teste post-hoc de Nemenyi condicionado ao teste de Friedman."""

import pandas as pd
import scikit_posthocs as sp
from .resultado_friedman import ResultadoFriedman
from .resultado_nemenyi import ResultadoNemenyi


class TesteNemenyi:
    """Executa comparações múltiplas de Nemenyi apenas quando Friedman for significativo."""

    __test__: bool = False

    def __init__(self, alpha: float = 0.05) -> None:
        self._alpha = alpha

    def executar(
        self,
        matriz_repeticoes: pd.DataFrame,
        resultado_friedman: ResultadoFriedman,
    ) -> ResultadoNemenyi:
        """Executa Nemenyi se Friedman for significativo.

        Parameters
        ----------
        matriz_repeticoes : pd.DataFrame
            Matriz de 30 repetições por modelo.
        resultado_friedman : ResultadoFriedman
            Resultado prévio do teste de Friedman.

        Returns
        -------
        ResultadoNemenyi
            Resultado do teste Nemenyi.
        """
        if not resultado_friedman.significativo or len(matriz_repeticoes.columns) < 2:
            return ResultadoNemenyi(
                matriz_p_valores=pd.DataFrame(),
                alpha=self._alpha,
                executado=False,
            )

        # sp.posthoc_nemenyi_friedman recebe a matriz (blocos x tratamentos)
        matriz_p = sp.posthoc_nemenyi_friedman(matriz_repeticoes.values)
        matriz_p.columns = matriz_repeticoes.columns
        matriz_p.index = matriz_repeticoes.columns

        return ResultadoNemenyi(
            matriz_p_valores=matriz_p,
            alpha=self._alpha,
            executado=True,
        )
