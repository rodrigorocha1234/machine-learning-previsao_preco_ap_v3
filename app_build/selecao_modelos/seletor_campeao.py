from collections.abc import Callable, Mapping
from typing import Final

from app_build.estatistica_modelos.contrato_estatistica import ResultadoFriedman
from app_build.selecao_modelos.contrato_seletor import ContratoSeletor, DecisaoSelecao
from app_build.selecao_modelos.ensemble_comite import EnsembleComite
from app_build.selecao_modelos.ranking_estatistico import RankingEstatistico
from app_build.validacao_cruzada.contrato_validador import ResultadoNestedCv


class SeletorCampeao:
    def __init__(self, usar_votacao: bool = True, quantidade_modelos: int = 3) -> None:
        fabrica_seletor: Final[Mapping[bool, Callable[[], ContratoSeletor]]] = {
            True: lambda: EnsembleComite(quantidade_modelos),
            False: RankingEstatistico,
        }
        self._seletor: Final[ContratoSeletor] = fabrica_seletor[usar_votacao]()

    def executar_selecao(
        self,
        resultados_cv: Mapping[str, ResultadoNestedCv],
        resultado_friedman: ResultadoFriedman,
    ) -> DecisaoSelecao:
        return self._seletor.selecionar(resultados_cv, resultado_friedman)
