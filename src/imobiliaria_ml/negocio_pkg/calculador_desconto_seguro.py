"""Calculador dinâmico de percentual de desconto seguro com base no erro do modelo."""

import numpy as np


class CalculadorDescontoSeguro:
    """Calcula o percentual de desconto que preserva a margem de segurança da imobiliária."""

    def __init__(self, margem_risco_percentual: float = 0.05) -> None:
        self._margem_risco = margem_risco_percentual

    def calcular_desconto_seguro(
        self,
        mae_validado: float,
        preco_medio: float,
        desconto_maximo: float = 10.0,
    ) -> float:
        """Determina o percentual de desconto seguro com base no erro típico e no preço médio.

        Parameters
        ----------
        mae_validado : float
            Erro Absoluto Médio do modelo em reais.
        preco_medio : float
            Preço médio de referência dos imóveis.
        desconto_maximo : float
            Teto máximo autorizado de desconto.

        Returns
        -------
        float
            Percentual sugerido de desconto seguro.
        """
        if preco_medio <= 0:
            return 0.0

        taxa_erro_modelo = mae_validado / preco_medio
        # Desconto seguro mantém folga contra o erro do modelo
        desconto_base = max(0.0, (self._margem_risco + taxa_erro_modelo / 2.0) * 100.0)
        return float(np.clip(desconto_base, 0.0, desconto_maximo))
