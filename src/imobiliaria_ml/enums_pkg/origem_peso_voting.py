"""Enumeração das origens de pesos para voting regressor."""

from enum import StrEnum


class OrigemPesoVoting(StrEnum):
    """Origens possíveis para o cálculo dos pesos de votação."""

    RMSE = "rmse"
    RANKING = "ranking"
