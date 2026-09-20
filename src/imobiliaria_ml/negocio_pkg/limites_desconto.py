"""Configuração de limites e alçadas para concessão de desconto."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LimitesDesconto:
    """Limites percentuais de desconto permitidos pela imobiliária."""

    minimo: float = 0.0
    maximo_automatico: float = 5.0
    maximo_com_aprovacao: float = 10.0
