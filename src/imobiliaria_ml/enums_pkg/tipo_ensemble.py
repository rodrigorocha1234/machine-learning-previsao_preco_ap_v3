"""Enumeração das estratégias de ensemble suportadas."""

from enum import StrEnum


class TipoEnsemble(StrEnum):
    """Técnicas de ensemble habilitadas pelo sistema."""

    VOTING_MEDIA = "voting_media"
    VOTING_PONDERADO_RMSE = "voting_ponderado_rmse"
    VOTING_PONDERADO_RANKING = "voting_ponderado_ranking"
    STACKING = "stacking"
    BAGGING = "bagging"
