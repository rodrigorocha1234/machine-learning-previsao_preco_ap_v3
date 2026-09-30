import time
from typing import override

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator, GridSearchCV

from app_build.ajuste_modelos.contrato_tuning import ContratoTuning, ResultadoTuning


class EstrategiaGrade(ContratoTuning):
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
        busca = GridSearchCV(
            estimator=estimador_base,
            param_grid=espaco_parametros,
            cv=validador_cv,
            scoring=metrica_scoring,
            n_jobs=None,
            refit=True,
        )
        busca.fit(dados_x, vetor_y)
        duracao = time.perf_counter() - inicio

        return ResultadoTuning(
            melhor_estimador=busca.best_estimator_,
            melhores_parametros=dict(busca.best_params_),
            melhor_score=float(busca.best_score_),
            tempo_execucao_segundos=duracao,
            grade_pesquisada=espaco_parametros,
            historico_resultados=pd.DataFrame(busca.cv_results_),
        )
