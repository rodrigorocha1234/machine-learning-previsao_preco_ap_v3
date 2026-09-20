"""Testes unitários para fábrica e estratégias de modelos."""

import numpy as np
import pytest
from imobiliaria_ml.enums_pkg.tipo_modelo import TipoModelo
from imobiliaria_ml.modelos_pkg.fabrica_modelos import FabricaModelos


@pytest.mark.parametrize(
    "tipo",
    [
        TipoModelo.REGRESSAO_LINEAR,
        TipoModelo.RIDGE,
        TipoModelo.LASSO,
        TipoModelo.ELASTIC_NET,
        TipoModelo.ARVORE_DECISAO,
        TipoModelo.RANDOM_FOREST,
        TipoModelo.GRADIENT_BOOSTING,
        TipoModelo.SVR,
        TipoModelo.REDE_NEURAL,
        TipoModelo.XGBOOST,
        TipoModelo.LIGHTGBM,
        TipoModelo.CATBOOST,
    ],
)
def test_fabrica_modelos_cria_e_ajusta(tipo: TipoModelo) -> None:
    fabrica = FabricaModelos()
    estrategia = fabrica.criar(tipo)
    assert estrategia.obter_tipo_modelo() == tipo

    modelo = estrategia.criar_modelo()
    assert hasattr(modelo, "fit")
    assert hasattr(modelo, "predict")

    # Teste de fit e predict com dados sintéticos simples
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([10.0, 20.0, 30.0, 40.0])

    modelo.fit(X, y)
    preds = modelo.predict(X)
    assert len(preds) == len(y)
