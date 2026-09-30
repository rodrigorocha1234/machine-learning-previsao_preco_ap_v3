from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricasRegressao:
    rmse: float
    mae: float
    mse: float
    r2: float
    rmse_relativo: float
    mape: float


@dataclass(frozen=True)
class ResultadoFoldExterno:
    indice_fold: int
    metricas: MetricasRegressao
    melhores_parametros: dict[str, object]
    previsoes: np.ndarray
    valores_reais: np.ndarray
    residuos: np.ndarray
    tempo_segundos: float


@dataclass(frozen=True)
class ResultadoNestedCv:
    nome_modelo: str
    resultados_folds: tuple[ResultadoFoldExterno, ...]
    metricas_medias: MetricasRegressao
    metricas_medianas: MetricasRegressao
    residuos_totais: np.ndarray


@runtime_checkable
class ContratoValidador(Protocol):
    def avaliar_modelo(
        self,
        nome_modelo: str,
        dados_x: pd.DataFrame,
        vetor_y: pd.Series,
    ) -> ResultadoNestedCv: ...
