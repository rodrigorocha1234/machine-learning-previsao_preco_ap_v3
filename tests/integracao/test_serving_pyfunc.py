"""Teste de integração para serving e inferência via PyFunc."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from imobiliaria_ml.preprocessamento_pkg.escalonador_standard import EscalonadorStandard
from imobiliaria_ml.preprocessamento_pkg.pre_processador import PreProcessador
from imobiliaria_ml.mlflow_pkg.modelo_imobiliario_pyfunc import ModeloImobiliarioPyFunc


def test_pyfunc_serving_inferencia_com_desconto(
    df_imoveis_sintetico: pd.DataFrame,
) -> None:
    """Testa o contrato exato do endpoint /invocations com PyFunc."""
    pre = PreProcessador(estrategia_escalonamento=EscalonadorStandard()).construir_pipeline(
        colunas_numericas=["Quartos", "Banheiros", "Vagas", "Metragem"],
        colunas_categoricas=["Zona"],
    )
    pipeline = Pipeline(steps=[("pre", pre), ("modelo", Ridge(alpha=1.0))])

    X = df_imoveis_sintetico.drop(columns=["Valor_da_Venda"])
    y = df_imoveis_sintetico["Valor_da_Venda"]
    pipeline.fit(X, y)

    pyfunc = ModeloImobiliarioPyFunc(
        modelo_pipeline=pipeline,
        desconto_maximo_permitido=10.0,
    )

    # Simulação da requisição que chega ao /invocations
    df_requisicao = pd.DataFrame(
        [
            {
                "Zona": "Centro",
                "Quartos": 3,
                "Banheiros": 2,
                "Vagas": 2,
                "Metragem": 120.0,
                "Percentual_Desconto": 5.0,
            }
        ]
    )

    resposta = pyfunc.predict(context=None, model_input=df_requisicao)

    assert "Valor_Previsto" in resposta.columns
    assert "Percentual_Desconto" in resposta.columns
    assert "Valor_Com_Desconto" in resposta.columns

    valor_previsto = resposta["Valor_Previsto"].iloc[0]
    desconto = resposta["Percentual_Desconto"].iloc[0]
    valor_com_desconto = resposta["Valor_Com_Desconto"].iloc[0]

    assert valor_previsto > 0
    assert desconto == 5.0
    assert np.isclose(valor_com_desconto, valor_previsto * 0.95)
