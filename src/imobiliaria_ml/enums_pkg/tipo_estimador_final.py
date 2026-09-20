"""Enumeração dos estimadores finais permitidos no Stacking."""

from enum import StrEnum


class TipoEstimadorFinal(StrEnum):
    """Estimadores finais suportados para meta-modelo de Stacking."""

    REGRESSAO_LINEAR = "regressao_linear"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
