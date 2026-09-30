from typing import ClassVar, Final

import numpy as np
from scipy import stats

from app_build.estatistica_modelos.contrato_estatistica import ResultadoShapiro


class TesteShapiro:
    __test__: ClassVar[bool] = False

    def __init__(
        self, limiar_alfa: float = 0.05, tamanho_maximo_amostra: int = 4000
    ) -> None:
        self._alfa: Final[float] = limiar_alfa
        self._tamanho_maximo: Final[int] = tamanho_maximo_amostra

    def testar(self, nome_modelo: str, residuos: np.ndarray) -> ResultadoShapiro:
        residuos_limpos = residuos[~np.isnan(residuos)]
        amostra = residuos_limpos[: self._tamanho_maximo]
        resultado = stats.shapiro(amostra)
        estatistica = float(resultado.statistic)
        p_valor = float(resultado.pvalue)
        eh_normal = bool(p_valor >= self._alfa)

        return ResultadoShapiro(
            nome_modelo=nome_modelo,
            estatistica=estatistica,
            p_valor=p_valor,
            eh_normal=eh_normal,
        )
