"""Enumeração dos eventos publicados ao Observer do MLflow."""

from enum import StrEnum


class TipoEvento(StrEnum):
    """Eventos disparados pelo pipeline durante o ciclo de Machine Learning."""

    EDA_FINALIZADA = "eda_finalizada"
    GRIDSEARCH_INICIADO = "gridsearch_iniciado"
    GRIDSEARCH_FINALIZADO = "gridsearch_finalizado"
    VALIDACAO_FINALIZADA = "validacao_finalizada"
    FRIEDMAN_FINALIZADO = "friedman_finalizado"
    NEMENYI_FINALIZADO = "nemenyi_finalizado"
    ENSEMBLE_FINALIZADO = "ensemble_finalizado"
    MODELO_CAMPEAO = "modelo_campeao"
    METRICAS_NEGOCIO_FINALIZADAS = "metricas_negocio_finalizadas"
    DRIFT_DETECTADO = "drift_detectado"
    CURVA_APRENDIZADO_GERADA = "curva_aprendizado_gerada"
