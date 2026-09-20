"""Regras de negócio de aplicação e validação de descontos comerciais."""

import numpy as np
import pandas as pd
from .limites_desconto import LimitesDesconto


class RegrasImobiliaria:
    """Implementa as regras de concessão de desconto e limites de alçada comercial."""

    def __init__(self, limites: LimitesDesconto | None = None) -> None:
        self._limites = limites or LimitesDesconto()

    def validar_desconto(self, percentual_desconto: float | pd.Series | np.ndarray) -> bool:
        """Verifica se o desconto solicitado está dentro dos limites operacionais.

        Parameters
        ----------
        percentual_desconto : float | pd.Series | np.ndarray
            Valor ou série percentual.

        Returns
        -------
        bool
            True se todos os descontos estiverem dentro dos limites permitidos.
        """
        arr = np.asarray(percentual_desconto, dtype=float)
        if (arr < self._limites.minimo).any() or (arr > self._limites.maximo_com_aprovacao).any():
            return False
        return True

    def calcular_valor_com_desconto(
        self,
        valor_previsto: float | np.ndarray | pd.Series,
        percentual_desconto: float | np.ndarray | pd.Series,
    ) -> float | np.ndarray:
        """Aplica a fórmula: Valor_Com_Desconto = Valor_Previsto * (1 - Percentual_Desconto / 100).

        Parameters
        ----------
        valor_previsto : float | np.ndarray | pd.Series
            Preço estimado pelo modelo de ML.
        percentual_desconto : float | np.ndarray | pd.Series
            Taxa percentual de desconto a aplicar.

        Returns
        -------
        float | np.ndarray
            Valor resultante pós-desconto.
        """
        if not self.validar_desconto(percentual_desconto):
            raise ValueError(
                f"Desconto fora da faixa permitida [{self._limites.minimo}%, {self._limites.maximo_com_aprovacao}%]."
            )

        fator = 1.0 - (np.asarray(percentual_desconto, dtype=float) / 100.0)
        resultado = np.asarray(valor_previsto, dtype=float) * fator

        if isinstance(valor_previsto, (float, int)) and isinstance(
            percentual_desconto, (float, int)
        ):
            return float(resultado.item())
        return resultado
