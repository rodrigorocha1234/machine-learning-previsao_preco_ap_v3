"""Calculador de métricas de regressão a partir de observações e predições."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from .metricas_regressao import MetricasRegressao


class CalculadorMetricas:
    """Calculador desacoplado para todas as métricas padrão de regressão."""

    def calcular(
        self,
        valores_reais: pd.Series | np.ndarray,
        previsoes: np.ndarray,
    ) -> MetricasRegressao:
        """Calcula RMSE, MAE, MSE, R2, MedAE e MAPE.

        Parameters
        ----------
        valores_reais : pd.Series | np.ndarray
            Alvos observados.
        previsoes : np.ndarray
            Valores preditos pelo estimador.

        Returns
        -------
        MetricasRegressao
            Objeto tipado com os scores calculados.
        """
        y_true = np.asarray(valores_reais, dtype=float)
        y_pred = np.asarray(previsoes, dtype=float)

        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        medae = float(median_absolute_error(y_true, y_pred))

        try:
            mape = float(mean_absolute_percentage_error(y_true, y_pred))
        except Exception:
            mape = float("nan")

        return MetricasRegressao(
            rmse=rmse,
            mae=mae,
            mse=mse,
            r2=r2,
            medae=medae,
            mape=mape,
        )
