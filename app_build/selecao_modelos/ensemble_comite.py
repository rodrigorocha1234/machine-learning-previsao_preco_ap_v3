from collections.abc import Mapping
from typing import Final, override

from app_build.estatistica_modelos.contrato_estatistica import ResultadoFriedman
from app_build.selecao_modelos.contrato_seletor import ContratoSeletor, DecisaoSelecao
from app_build.validacao_cruzada.contrato_validador import ResultadoNestedCv


class EnsembleComite(ContratoSeletor):
    def __init__(self, quantidade_modelos: int = 3) -> None:
        self._qtd: Final[int] = quantidade_modelos

    @override
    def selecionar(
        self,
        resultados_cv: Mapping[str, ResultadoNestedCv],
        resultado_friedman: ResultadoFriedman,
    ) -> DecisaoSelecao:
        ranks = resultado_friedman.ranks_medios
        modelos_ordenados = sorted(
            ranks.keys(),
            key=lambda nome: (ranks[nome], resultados_cv[nome].metricas_medianas.rmse),
        )
        top_k = tuple(modelos_ordenados[: self._qtd])
        campeao_lider = top_k[0]

        justificativa = (
            f"Comite de votacao formado pelos top {len(top_k)} modelos: {top_k}. "
            f"Lider do comite: '{campeao_lider}' com rank {ranks[campeao_lider]:.2f}."
        )

        return DecisaoSelecao(
            modelo_principal=campeao_lider,
            modelos_selecionados=top_k,
            modo_selecao="ENSEMBLE_VOTACAO",
            justificativa=justificativa,
        )
