from collections.abc import Mapping

import pandas as pd

from app_build.validacao_cruzada.contrato_validador import ResultadoNestedCv


class ExtratorResiduos:
    @staticmethod
    def extrair_matriz_rmse(
        resultados: Mapping[str, ResultadoNestedCv],
    ) -> pd.DataFrame:
        dados_colunas: dict[str, list[float]] = {}
        for nome_modelo, res_cv in resultados.items():
            dados_colunas[nome_modelo] = [
                fold.metricas.rmse for fold in res_cv.resultados_folds
            ]
        return pd.DataFrame(dados_colunas)

    @staticmethod
    def extrair_matriz_mae(resultados: Mapping[str, ResultadoNestedCv]) -> pd.DataFrame:
        dados_colunas: dict[str, list[float]] = {}
        for nome_modelo, res_cv in resultados.items():
            dados_colunas[nome_modelo] = [
                fold.metricas.mae for fold in res_cv.resultados_folds
            ]
        return pd.DataFrame(dados_colunas)
