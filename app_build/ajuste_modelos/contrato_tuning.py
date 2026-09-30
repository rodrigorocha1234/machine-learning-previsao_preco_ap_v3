from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator


@dataclass(frozen=True)
class ResultadoTuning:
    melhor_estimador: BaseEstimator
    melhores_parametros: dict[str, object]
    melhor_score: float
    tempo_execucao_segundos: float
    grade_pesquisada: dict[str, object]
    historico_resultados: pd.DataFrame


@runtime_checkable
class ContratoTuning(Protocol):
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
    ) -> ResultadoTuning: ...
