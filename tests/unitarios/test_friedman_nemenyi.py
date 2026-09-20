"""Testes unitários para testes estatísticos não-paramétricos e grupo elegível."""

import numpy as np
import pandas as pd
from imobiliaria_ml.validacao_pkg.teste_friedman import TesteFriedman
from imobiliaria_ml.validacao_pkg.teste_nemenyi import TesteNemenyi
from imobiliaria_ml.validacao_pkg.seletor_campeao import SeletorCampeao
from imobiliaria_ml.validacao_pkg.resultado_validacao_cruzada import ResultadoValidacaoCruzada


def test_teste_friedman_calcula_p_valor() -> None:
    np.random.seed(42)
    # Matriz com 30 repetições e 3 modelos com desempenhos distintos
    m1 = np.random.normal(50.0, 2.0, size=30)
    m2 = np.random.normal(70.0, 2.0, size=30)
    m3 = np.random.normal(90.0, 2.0, size=30)
    df_matriz = pd.DataFrame({"modelo_a": m1, "modelo_b": m2, "modelo_c": m3})

    teste = TesteFriedman(alpha=0.05)
    res = teste.executar(df_matriz)

    assert res.significativo
    assert res.p_valor < 0.05
    assert res.rankings_medios["modelo_a"] < res.rankings_medios["modelo_c"]


def test_teste_nemenyi_executado_quando_friedman_significativo() -> None:
    np.random.seed(42)
    m1 = np.random.normal(50.0, 2.0, size=30)
    m2 = np.random.normal(70.0, 2.0, size=30)
    m3 = np.random.normal(90.0, 2.0, size=30)
    df_matriz = pd.DataFrame({"modelo_a": m1, "modelo_b": m2, "modelo_c": m3})

    friedman = TesteFriedman().executar(df_matriz)
    nemenyi = TesteNemenyi().executar(df_matriz, friedman)

    assert nemenyi.executado
    assert not nemenyi.matriz_p_valores.empty
    assert nemenyi.matriz_p_valores.shape == (3, 3)


def test_seletor_campeao_forma_grupo_elegivel() -> None:
    np.random.seed(42)
    m1 = np.random.normal(50.0, 1.0, size=30)
    m2 = np.random.normal(50.5, 1.0, size=30)  # Muito similar a m1
    m3 = np.random.normal(90.0, 1.0, size=30)  # Bem pior

    df_matriz = pd.DataFrame({"m1": m1, "m2": m2, "m3": m3})
    res_cv = ResultadoValidacaoCruzada(matriz_repeticoes=df_matriz)

    friedman = TesteFriedman().executar(df_matriz)
    nemenyi = TesteNemenyi().executar(df_matriz, friedman)

    seletor = SeletorCampeao()
    grupo = seletor.selecionar_grupo(res_cv, friedman, nemenyi)

    assert grupo.melhor_modelo == "m1"
    assert "m1" in grupo.modelos_elegiveis
    assert "m2" in grupo.modelos_elegiveis
    assert "m3" not in grupo.modelos_elegiveis
