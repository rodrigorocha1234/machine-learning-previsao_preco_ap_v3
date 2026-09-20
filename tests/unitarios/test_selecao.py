"""Testes unitários para seleção de hiperparâmetros com GridSearchCV."""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imobiliaria_ml.enums_pkg.tipo_modelo import TipoModelo
from imobiliaria_ml.modelos_pkg.fabrica_modelos import FabricaModelos
from imobiliaria_ml.selecao_pkg.seletor_hiperparametros import SeletorHiperparametros


def test_seletor_hiperparametros_otimiza(df_imoveis_sintetico: pd.DataFrame) -> None:
    fabrica = FabricaModelos()
    estrategia = fabrica.criar(TipoModelo.RIDGE)

    X = df_imoveis_sintetico[["Quartos", "Banheiros", "Metragem"]]
    y = df_imoveis_sintetico["Valor_da_Venda"]

    pipeline = Pipeline(
        steps=[("scaler", StandardScaler()), ("modelo", estrategia.criar_modelo())]
    )

    seletor = SeletorHiperparametros(cv_splits=3)
    grade = {"modelo__alpha": [0.1, 1.0]}

    resultado = seletor.otimizar(
        estrategia=estrategia,
        pipeline_base=pipeline,
        X_treino=X,
        y_treino=y,
        grade_customizada=grade,
    )

    assert resultado.tipo_modelo == TipoModelo.RIDGE
    assert "modelo__alpha" in resultado.melhores_parametros
    assert resultado.melhor_score_rmse > 0
    assert not resultado.tabela_cv_results.empty
