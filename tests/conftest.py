"""Fixtures globais para testes unitários e de integração."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def df_imoveis_sintetico() -> pd.DataFrame:
    """Gera um DataFrame sintético representativo para testes rápidos."""
    np.random.seed(42)
    n = 60
    zonas = ["Zona Sul", "Zona Norte", "Centro", "Zona Leste", "Zona Oeste"]

    zonas_col = [zonas[i % len(zonas)] for i in range(n)]
    quartos = np.random.randint(1, 5, size=n)
    banheiros = np.random.randint(1, 5, size=n)
    vagas = np.random.randint(0, 4, size=n)
    metragem = np.random.uniform(40.0, 250.0, size=n)

    # Fórmula com ruído para simular o alvo
    valor = (
        80_000.0
        + 40_000.0 * quartos
        + 25_000.0 * banheiros
        + 30_000.0 * vagas
        + 3_500.0 * metragem
        + np.random.normal(0, 15_000.0, size=n)
    )
    valor = np.maximum(valor, 100_000.0)

    return pd.DataFrame(
        {
            "Zona": zonas_col,
            "Quartos": quartos,
            "Banheiros": banheiros,
            "Vagas": vagas,
            "Metragem": metragem,
            "Valor_da_Venda": valor,
        }
    )
