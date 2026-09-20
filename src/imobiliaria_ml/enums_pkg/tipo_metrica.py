"""Enumeração das métricas de avaliação de regressão."""

from enum import StrEnum


class TipoMetrica(StrEnum):
    """Métricas numéricas de desempenho em regressão."""

    RMSE = "rmse"
    MAE = "mae"
    MSE = "mse"
    R2 = "r2"
    MEDAE = "medae"
    MAPE = "mape"
