"""Testes unitários para métricas financeiras e de negócio imobiliário."""

import numpy as np
import pandas as pd
from imobiliaria_ml.negocio_pkg.calculador_metricas_imobiliaria import CalculadorMetricasImobiliaria
from imobiliaria_ml.negocio_pkg.calculador_desconto_seguro import CalculadorDescontoSeguro


def test_calculador_desconto_seguro() -> None:
    calc = CalculadorDescontoSeguro(margem_risco_percentual=0.05)
    desconto = calc.calcular_desconto_seguro(mae_validado=20_000.0, preco_medio=500_000.0)
    assert 0.0 <= desconto <= 10.0


def test_calculador_metricas_imobiliaria() -> None:
    calc = CalculadorMetricasImobiliaria()
    X = pd.DataFrame({"Zona": ["Centro", "Zona Sul", "Centro"]})
    y_real = pd.Series([500_000.0, 600_000.0, 700_000.0])
    y_pred = np.array([510_000.0, 590_000.0, 720_000.0])

    res = calc.calcular_metricas(X, y_real, y_pred)
    assert res.cobertura_5 > 0.5
    assert res.cobertura_10 == 1.0
    assert not res.tabela_erro_por_faixa.empty
    assert not res.tabela_erro_por_zona.empty
