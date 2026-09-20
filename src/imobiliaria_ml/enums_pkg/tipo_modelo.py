"""Enumeração de tipos de modelos de regressão."""

from enum import StrEnum


class TipoModelo(StrEnum):
    """Tipos de modelos suportados pela solução de ML."""

    REGRESSAO_LINEAR = "regressao_linear"
    REGRESSAO_MULTIPLA = "regressao_multipla"
    REGRESSAO_POLINOMIAL = "regressao_polinomial"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    ARVORE_DECISAO = "arvore_decisao"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    SVR = "svr"
    REDE_NEURAL = "rede_neural"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
