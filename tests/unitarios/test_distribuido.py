"""Testes unitários para executor e adaptador distribuído."""

import numpy as np
from sklearn.linear_model import LinearRegression
from imobiliaria_ml.distribuido_pkg.executor_sklearn import ExecutorSklearn
from imobiliaria_ml.distribuido_pkg.adaptador_distribuido import AdaptadorDistribuido


def test_executor_sklearn_ajusta_e_preve() -> None:
    executor = ExecutorSklearn()
    modelo = LinearRegression()
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])

    modelo_ajustado = executor.ajustar(modelo, X, y)
    preds = executor.prever(modelo_ajustado, X)

    assert len(preds) == 3
    assert np.isclose(preds[0], 2.0)


def test_adaptador_distribuido_adaptar() -> None:
    adaptador = AdaptadorDistribuido()
    modelo = LinearRegression()
    modelo_adaptado = adaptador.adaptar(modelo)
    assert modelo_adaptado is modelo
