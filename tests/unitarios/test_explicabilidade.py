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


def test_extrator_gerar_equacao_txt_linear() -> None:
    from sklearn.linear_model import Ridge

    X = pd.DataFrame({"Metragem": [50.0, 100.0, 150.0], "Quartos": [1, 2, 3]})
    y = pd.Series([150_000.0, 300_000.0, 450_000.0])

    pipeline = Pipeline(steps=[("pre", StandardScaler()), ("modelo", Ridge(alpha=1.0))])
    pipeline.fit(X, y)

    extrator = ExtratorCoeficientes()
    txt = extrator.gerar_equacao_txt(pipeline, nome_modelo="ridge")

    assert "EQUAÇÃO DO MODELO: RIDGE" in txt
    assert "Valor_da_Venda" in txt
    assert "INTERCEPTO BASE" in txt
    assert "COEFICIENTES DAS VARIÁVEIS" in txt


def test_extrator_gerar_equacao_txt_arvore() -> None:
    from sklearn.tree import DecisionTreeRegressor

    X = pd.DataFrame({"Metragem": [50.0, 100.0, 150.0], "Quartos": [1, 2, 3]})
    y = pd.Series([150_000.0, 300_000.0, 450_000.0])

    pipeline = Pipeline(steps=[("pre", StandardScaler()), ("modelo", DecisionTreeRegressor(max_depth=3))])
    pipeline.fit(X, y)

    extrator = ExtratorCoeficientes()
    txt = extrator.gerar_equacao_txt(pipeline, nome_modelo="arvore_decisao")

    assert "EQUAÇÃO DO MODELO: ARVORE_DECISAO" in txt
    assert "Árvore de Decisão" in txt
    assert "Valor_da_Venda = ∑_{m=1}^{M}" in txt
    assert "REGRAS DE DECISÃO" in txt


def test_extrator_gerar_equacao_txt_ensemble_e_boosting() -> None:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    X = pd.DataFrame({"Metragem": [50.0, 100.0, 150.0, 200.0], "Quartos": [1, 2, 3, 4]})
    y = pd.Series([150_000.0, 300_000.0, 450_000.0, 600_000.0])

    pipe_rf = Pipeline(steps=[("pre", StandardScaler()), ("modelo", RandomForestRegressor(n_estimators=10, random_state=42))])
    pipe_rf.fit(X, y)

    extrator = ExtratorCoeficientes()
    txt_rf = extrator.gerar_equacao_txt(pipe_rf, nome_modelo="random_forest")
    assert "EQUAÇÃO DO MODELO: RANDOM_FOREST" in txt_rf
    assert "Valor_da_Venda = (1 / B) * ∑_{b=1}^{B} T_b(X)" in txt_rf

    pipe_gb = Pipeline(steps=[("pre", StandardScaler()), ("modelo", GradientBoostingRegressor(n_estimators=10, random_state=42))])
    pipe_gb.fit(X, y)

    txt_gb = extrator.gerar_equacao_txt(pipe_gb, nome_modelo="gradient_boosting")
    assert "EQUAÇÃO DO MODELO: GRADIENT_BOOSTING" in txt_gb
    assert "Valor_da_Venda = F₀(X) + η * ∑_{m=1}^{M} h_m(X)" in txt_gb

