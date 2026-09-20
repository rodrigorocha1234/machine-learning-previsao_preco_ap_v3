"""Dataclass estruturada com as métricas técnicas de regressão."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricasRegressao:
    """Conjunto de métricas de avaliação de modelos de regressão."""

    rmse: float
    mae: float
    mse: float
    r2: float
    medae: float
    mape: float
