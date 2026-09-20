"""Enumeração dos níveis de severidade de data drift."""

from enum import StrEnum


class NivelDrift(StrEnum):
    """Níveis de severidade detectados no monitoramento de drift."""

    ESTAVEL = "estavel"
    ATENCAO = "atencao"
    FORTE = "forte"
