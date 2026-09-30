from collections.abc import Mapping
from typing import override

from app_build.estatistica_modelos.contrato_estatistica import ResultadoFriedman
from app_build.selecao_modelos.contrato_seletor import ContratoSeletor, DecisaoSelecao
from app_build.validacao_cruzada.contrato_validador import ResultadoNestedCv


class RankingEstatistico(ContratoSeletor):
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
        campeao = modelos_ordenados[0]
        rank_campeao = ranks[campeao]
        rmse_campeao = resultados_cv[campeao].metricas_medianas.rmse

        justificativa = (
            f"Modelo '{campeao}' selecionado como campeao unico por possuir menor rank medio no Friedman "
            f"({rank_campeao:.2f}) e RMSE mediano de {rmse_campeao:.2f} nos folds externos."
        )

        return DecisaoSelecao(
            modelo_principal=campeao,
            modelos_selecionados=(campeao,),
            modo_selecao="MODELO_UNICO",
            justificativa=justificativa,
        )
