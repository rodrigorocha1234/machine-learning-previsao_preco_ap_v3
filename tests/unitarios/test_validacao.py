"""Testes unitários para cálculo de métricas e RepeatedKFold."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imobiliaria_ml.validacao_pkg.calculador_metricas import CalculadorMetricas
from imobiliaria_ml.validacao_pkg.validador_repeated_kfold import ValidadorRepeatedKFold


def test_calculador_metricas_precisao() -> None:
    calc = CalculadorMetricas()
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])

    metricas = calc.calcular(y_true, y_pred)
    assert metricas.mae == 10.0
    assert metricas.rmse == 10.0
    assert metricas.mse == 100.0
    assert metricas.r2 > 0.95


def test_validador_repeated_kfold_particoes(df_imoveis_sintetico: pd.DataFrame) -> None:
    # Teste rápido com 3 splits e 2 repetições
    validador = ValidadorRepeatedKFold(n_splits=3, n_repeats=2, random_state=42)

    X = df_imoveis_sintetico[["Quartos", "Banheiros", "Metragem"]]
    y = df_imoveis_sintetico["Valor_da_Venda"]

    pipeline = Pipeline(steps=[("scaler", StandardScaler()), ("modelo", Ridge(alpha=1.0))])
    res = validador.validar_modelos({"ridge": pipeline}, X, y)

    assert len(res.tabela_folds) == 6  # 3 splits * 2 repeats
    assert res.matriz_repeticoes.shape == (2, 1)  # 2 repeats x 1 model
