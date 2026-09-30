import itertools
import math
from typing import ClassVar, Final

from app_build.estatistica_modelos.contrato_estatistica import (
    ComparacaoParNemenyi,
    ResultadoFriedman,
    ResultadoNemenyi,
)


class TesteNemenyi:
    __test__: ClassVar[bool] = False

    Q_ALFA_TABELA: Final[dict[int, float]] = {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
    }

    def calcular_cd(self, total_modelos: int, total_folds: int) -> float:
        q_alfa = self.Q_ALFA_TABELA.get(total_modelos, 3.164)
        fator = (total_modelos * (total_modelos + 1.0)) / (6.0 * total_folds)
        return float(q_alfa * math.sqrt(fator))

    def testar(
        self, resultado_friedman: ResultadoFriedman, total_folds: int
    ) -> ResultadoNemenyi:
        ranks = resultado_friedman.ranks_medios
        modelos = list(ranks.keys())
        total_modelos = len(modelos)
        cd = self.calcular_cd(total_modelos, total_folds)

        def gerar_comparacao(par: tuple[str, str]) -> ComparacaoParNemenyi:
            mod_a, mod_b = par
            dif = abs(ranks[mod_a] - ranks[mod_b])
            eh_sig = bool(resultado_friedman.eh_significativo and (dif > cd))
            return ComparacaoParNemenyi(
                modelo_a=mod_a,
                modelo_b=mod_b,
                diferenca_ranks=dif,
                diferenca_critica=cd,
                eh_significativo=eh_sig,
            )

        pares = list(itertools.combinations(modelos, 2))
        comparacoes = tuple(map(gerar_comparacao, pares))

        return ResultadoNemenyi(
            diferenca_critica=cd,
            comparacoes=comparacoes,
        )
