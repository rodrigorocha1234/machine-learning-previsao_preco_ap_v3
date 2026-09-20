"""Execução do teste de Friedman sobre os blocos de repetições da validação."""

import pandas as pd
from scipy.stats import friedmanchisquare
from .resultado_friedman import ResultadoFriedman


class TesteFriedman:
    """Aplica o teste de Friedman nas 30 repetições agregadas da validação cruzada."""

    __test__: bool = False

    def __init__(self, alpha: float = 0.05) -> None:
        self._alpha = alpha

    def executar(self, matriz_repeticoes: pd.DataFrame) -> ResultadoFriedman:
        """Executa o teste de Friedman sobre a matriz de repetições.

        Parameters
        ----------
        matriz_repeticoes : pd.DataFrame
            DataFrame onde cada linha é uma repetição e cada coluna é um modelo (valores de RMSE).

        Returns
        -------
        ResultadoFriedman
            Resultado estatístico do teste.
        """
        colunas = list(matriz_repeticoes.columns)
        if len(colunas) < 3:
            # Friedman exige no mínimo 3 tratamentos (amostras pareadas)
            rankings_iniciais = {
                col: float(i + 1)
                for i, col in enumerate(
                    matriz_repeticoes.mean().sort_values().index
                )
            } if colunas else {}
            return ResultadoFriedman(
                estatistica=0.0,
                p_valor=1.0,
                alpha=self._alpha,
                significativo=False,
                rankings_medios=rankings_iniciais,
            )

        # Cálculo do ranking em cada bloco (menor RMSE = ranking 1)
        ranks_por_linha = matriz_repeticoes.rank(axis=1, ascending=True)
        rankings_medios_serie = ranks_por_linha.mean(axis=0)
        rankings_medios: dict[str, float] = {
            col: float(rankings_medios_serie[col]) for col in colunas
        }

        # Execução do teste de Friedman
        amostras = [matriz_repeticoes[col].values for col in colunas]
        resultado_scipy = friedmanchisquare(*amostras)

        estatistica = float(resultado_scipy.statistic)
        p_valor = float(resultado_scipy.pvalue)
        significativo = bool(p_valor < self._alpha)

        return ResultadoFriedman(
            estatistica=estatistica,
            p_valor=p_valor,
            alpha=self._alpha,
            significativo=significativo,
            rankings_medios=rankings_medios,
        )
