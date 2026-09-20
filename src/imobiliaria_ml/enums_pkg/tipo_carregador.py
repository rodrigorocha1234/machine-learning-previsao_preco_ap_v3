"""Enumeração dos tipos de carregadores de dados."""

from enum import StrEnum


class TipoCarregador(StrEnum):
    """Tipos de fontes de dados suportadas."""

    CSV = "csv"
    EXCEL = "excel"
    BANCO = "banco"
