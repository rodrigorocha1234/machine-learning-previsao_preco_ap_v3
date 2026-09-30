import time
from typing import override

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator

from app_build.ajuste_modelos.contrato_tuning import ContratoTuning, ResultadoTuning


class EstrategiaNula(ContratoTuning):
    @override
    def executar_tuning(
        self,
        estimador_base: BaseEstimator,
        espaco_parametros: dict[str, object],
        dados_x: pd.DataFrame,
        vetor_y: pd.Series,
        validador_cv: BaseCrossValidator,
        metrica_scoring: str,
        semente: int,
        iteracoes: int,
    ) -> ResultadoTuning:
        inicio = time.perf_counter()
        estimador_base.fit(dados_x, vetor_y)
        duracao = time.perf_counter() - inicio

        return ResultadoTuning(
            melhor_estimador=estimador_base,
            melhores_parametros={},
            melhor_score=0.0,
            tempo_execucao_segundos=duracao,
            grade_pesquisada={},
            historico_resultados=pd.DataFrame(),
        )
