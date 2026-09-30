import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from app_build.validacao_cruzada.contrato_validador import MetricasRegressao


class AcumuladorMetricas:
    @staticmethod
    def calcular_metricas(
        valores_reais: np.ndarray, previsoes: np.ndarray
    ) -> MetricasRegressao:
        mse = float(mean_squared_error(valores_reais, previsoes))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(valores_reais, previsoes))
        r2 = float(r2_score(valores_reais, previsoes))
        mape = float(mean_absolute_percentage_error(valores_reais, previsoes))

        media_real = float(np.mean(valores_reais))
        rmse_relativo = float(rmse / media_real) if media_real > 0 else 0.0

        return MetricasRegressao(
            rmse=rmse,
            mae=mae,
            mse=mse,
            r2=r2,
            rmse_relativo=rmse_relativo,
            mape=mape,
        )

    @staticmethod
    def agregar_medias(
        metricas_lista: tuple[MetricasRegressao, ...],
    ) -> MetricasRegressao:
        return MetricasRegressao(
            rmse=float(np.mean([m.rmse for m in metricas_lista])),
            mae=float(np.mean([m.mae for m in metricas_lista])),
            mse=float(np.mean([m.mse for m in metricas_lista])),
            r2=float(np.mean([m.r2 for m in metricas_lista])),
            rmse_relativo=float(np.mean([m.rmse_relativo for m in metricas_lista])),
            mape=float(np.mean([m.mape for m in metricas_lista])),
        )

    @staticmethod
    def agregar_medianas(
        metricas_lista: tuple[MetricasRegressao, ...],
    ) -> MetricasRegressao:
        return MetricasRegressao(
            rmse=float(np.median([m.rmse for m in metricas_lista])),
            mae=float(np.median([m.mae for m in metricas_lista])),
            mse=float(np.median([m.mse for m in metricas_lista])),
            r2=float(np.median([m.r2 for m in metricas_lista])),
            rmse_relativo=float(np.median([m.rmse_relativo for m in metricas_lista])),
            mape=float(np.median([m.mape for m in metricas_lista])),
        )
