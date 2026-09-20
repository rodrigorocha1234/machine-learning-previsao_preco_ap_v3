"""Testes unitários para técnicas de ensemble (Voting, Stacking, Bagging)."""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from imobiliaria_ml.ensemble_pkg.calculador_pesos_rmse import CalculadorPesosRmse
from imobiliaria_ml.ensemble_pkg.calculador_pesos_ranking import CalculadorPesosRanking
from imobiliaria_ml.ensemble_pkg.votacao_regressor import VotacaoRegressor
from imobiliaria_ml.ensemble_pkg.votacao_regressor_ponderada import VotacaoRegressorPonderada
from imobiliaria_ml.ensemble_pkg.stacking_regressor_imobiliario import StackingRegressorImobiliario
from imobiliaria_ml.ensemble_pkg.bagging_regressor_imobiliario import BaggingRegressorImobiliario


def test_calculador_pesos_rmse() -> None:
    calc = CalculadorPesosRmse()
    # Modelo com menor RMSE deve receber maior peso
    pesos = calc.calcular({"m1": 10.0, "m2": 20.0})
    assert pesos["m1"] > pesos["m2"]
    assert np.isclose(sum(pesos.values()), 1.0)


def test_calculador_pesos_ranking() -> None:
    calc = CalculadorPesosRanking()
    # Modelo com ranking 1 deve receber maior peso do que modelo com ranking 2
    pesos = calc.calcular({"m1": 1.0, "m2": 2.0})
    assert pesos["m1"] > pesos["m2"]
    assert np.isclose(sum(pesos.values()), 1.0)


def test_voting_regressor_executa() -> None:
    m1 = LinearRegression()
    m2 = Ridge(alpha=1.0)
    estrategia = VotacaoRegressor()
    ensemble = estrategia.criar_estimador([("m1", m1), ("m2", m2)])

    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    ensemble.fit(X, y)
    preds = ensemble.predict(X)
    assert len(preds) == 4


def test_stacking_regressor_executa() -> None:
    m1 = LinearRegression()
    m2 = Ridge(alpha=1.0)
    estrategia = StackingRegressorImobiliario()
    stacking = estrategia.criar_estimador([("m1", m1), ("m2", m2)])

    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    stacking.fit(X, y)
    preds = stacking.predict(X)
    assert len(preds) == 6


def test_bagging_regressor_executa() -> None:
    estrategia = BaggingRegressorImobiliario(n_estimators=10)
    bagging = estrategia.criar_estimador([("base", LinearRegression())])

    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    bagging.fit(X, y)
    preds = bagging.predict(X)
    assert len(preds) == 4
