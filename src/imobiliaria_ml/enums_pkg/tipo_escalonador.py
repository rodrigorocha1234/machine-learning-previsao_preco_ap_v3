"""Enumeração dos tipos de escalonadores numéricos."""

from enum import StrEnum


class TipoEscalonador(StrEnum):
    """Estratégias de escalonamento suportadas no pré-processamento."""

    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUSTO = "robusto"
    SEM_ESCALA = "sem_escala"
