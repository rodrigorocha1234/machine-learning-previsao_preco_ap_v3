from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class SimulacaoFinanceira:
    desconto_moderado_5: float
    desconto_agressivo_10: float
    desconto_queima_15: float
    faixa_segura_piso: float
    faixa_segura_teto: float


class SimuladorDescontos:
    def __init__(self, margem_seguranca_padrao: float = 0.10) -> None:
        self._margem_segura: Final[float] = margem_seguranca_padrao

    def simular(self, valor_previsto: float) -> SimulacaoFinanceira:
        desconto_5 = float(valor_previsto * 0.95)
        desconto_10 = float(valor_previsto * 0.90)
        desconto_15 = float(valor_previsto * 0.85)
        piso = float(valor_previsto * (1.0 - self._margem_segura))
        teto = float(valor_previsto)

        return SimulacaoFinanceira(
            desconto_moderado_5=desconto_5,
            desconto_agressivo_10=desconto_10,
            desconto_queima_15=desconto_15,
            faixa_segura_piso=piso,
            faixa_segura_teto=teto,
        )
