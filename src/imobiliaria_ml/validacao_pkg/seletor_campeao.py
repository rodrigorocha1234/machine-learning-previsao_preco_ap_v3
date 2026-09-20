"""Seletor do grupo estatisticamente elegível e do modelo campeão individual."""

import pandas as pd
from .resultado_friedman import ResultadoFriedman
from .resultado_nemenyi import ResultadoNemenyi
from .resultado_validacao_cruzada import ResultadoValidacaoCruzada
from .grupo_elegivel import GrupoElegivel


class SeletorCampeao:
    """Forma o grupo de modelos estatisticamente elegíveis e seleciona o campeão."""

    def selecionar_grupo(
        self,
        resultado_cv: ResultadoValidacaoCruzada,
        resultado_friedman: ResultadoFriedman,
        resultado_nemenyi: ResultadoNemenyi,
    ) -> GrupoElegivel:
        """Determina os modelos elegíveis a partir dos testes de Friedman e Nemenyi.

        Parameters
        ----------
        resultado_cv : ResultadoValidacaoCruzada
            Resultados consolidados da validação cruzada.
        resultado_friedman : ResultadoFriedman
            Resultados do teste de Friedman.
        resultado_nemenyi : ResultadoNemenyi
            Resultados do teste de Nemenyi.

        Returns
        -------
        GrupoElegivel
            Grupo consolidado com o melhor modelo e os estatisticamente equivalentes.
        """
        modelos = list(resultado_cv.matriz_repeticoes.columns)
        if not modelos:
            return GrupoElegivel(melhor_modelo="", modelos_elegiveis=())

        # Identifica médias de RMSE
        rmse_medios: dict[str, float] = {
            col: float(resultado_cv.matriz_repeticoes[col].mean()) for col in modelos
        }

        # Identifica o melhor ranking médio (menor valor)
        rankings = resultado_friedman.rankings_medios or {m: 1.0 for m in modelos}
        melhor_modelo = min(rankings.keys(), key=lambda m: rankings[m])

        elegiveis: list[str] = [melhor_modelo]

        if resultado_friedman.significativo and resultado_nemenyi.executado:
            # Inclui modelos cuja diferença para o melhor modelo NÃO seja significativa (p >= alpha)
            p_valores = resultado_nemenyi.matriz_p_valores
            for m in modelos:
                if m != melhor_modelo and m in p_valores.columns and melhor_modelo in p_valores.index:
                    p = float(p_valores.loc[m, melhor_modelo])
                    if p >= resultado_nemenyi.alpha:
                        elegiveis.append(m)
        else:
            # Sem diferença estatística significativa global: todos permanecem elegíveis
            elegiveis = list(modelos)

        # Ordena elegíveis pelo menor RMSE médio
        elegiveis.sort(key=lambda m: rmse_medios[m])

        return GrupoElegivel(
            melhor_modelo=elegiveis[0],
            modelos_elegiveis=tuple(elegiveis),
            rankings=rankings,
            rmse_medios=rmse_medios,
        )
