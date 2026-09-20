"""Testes unitários para explicabilidade e extração de coeficientes."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imobiliaria_ml.explicabilidade_pkg.extrator_coeficientes import ExtratorCoeficientes
from imobiliaria_ml.explicabilidade_pkg.interpretador_modelo import InterpretadorModelo


def test_extrator_coeficientes_gera_equacao() -> None:
    X = pd.DataFrame({"Metragem": [50.0, 100.0, 150.0], "Quartos": [1, 2, 3]})
    y = pd.Series([150_000.0, 300_000.0, 450_000.0])

    pipeline = Pipeline(steps=[("pre", StandardScaler()), ("modelo", LinearRegression())])
    pipeline.fit(X, y)

    extrator = ExtratorCoeficientes()
    df_coef, intercepto, equacao = extrator.extrair(pipeline)

    assert len(df_coef) == 2
    assert "Valor_da_Venda" in equacao
    assert isinstance(intercepto, float)

    interpretador = InterpretadorModelo()
    md = interpretador.gerar_interpretacao_markdown(df_coef, intercepto)
    assert "Interpretação de Negócio" in md
