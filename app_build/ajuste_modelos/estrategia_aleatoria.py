import time
from typing import override

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator, RandomizedSearchCV

from app_build.ajuste_modelos.contrato_tuning import ContratoTuning, ResultadoTuning


class EstrategiaAleatoria(ContratoTuning):
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
        busca = RandomizedSearchCV(
            estimator=estimador_base,
            param_distributions=espaco_parametros,
            n_iter=iteracoes,
            cv=validador_cv,
            scoring=metrica_scoring,
            random_state=semente,
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
