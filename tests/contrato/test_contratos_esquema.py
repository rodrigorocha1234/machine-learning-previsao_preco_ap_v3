"""Testes de contrato para esquemas de entrada, saída e limites de regras."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from imobiliaria_ml.qualidade_pkg.validador_esquema import ValidadorEsquema
from imobiliaria_ml.negocio_pkg.regras_imobiliaria import RegrasImobiliaria
from imobiliaria_ml.negocio_pkg.limites_desconto import LimitesDesconto
from imobiliaria_ml.mlflow_pkg.modelo_imobiliario_pyfunc import ModeloImobiliarioPyFunc


def test_esquema_rejeita_coluna_faltante() -> None:
    """Valida que ausência de coluna obrigatória gera erro."""
    validador = ValidadorEsquema()
    df_invalido = pd.DataFrame({"Zona": ["Centro"], "Quartos": [2]})
    res = validador.validar(df_invalido, exigir_alvo=True)
    assert not res.valido
    assert any("Coluna obrigatória ausente" in err for err in res.mensagens_erro)


def test_esquema_rejeita_metragem_negativa_ou_zero() -> None:
    """Valida que metragem <= 0 é invalidada."""
    validador = ValidadorEsquema()
    df = pd.DataFrame(
        {
            "Zona": ["Centro"],
            "Quartos": [2],
            "Banheiros": [1],
            "Vagas": [1],
            "Metragem": [0.0],
            "Valor_da_Venda": [300_000.0],
        }
    )
    res = validador.validar(df, exigir_alvo=True)
    assert not res.valido


def test_regras_imobiliaria_valida_desconto_permitido() -> None:
    """Valida aplicação de limites de desconto pela regra de negócio."""
    regras = RegrasImobiliaria(LimitesDesconto(minimo=0.0, maximo_com_aprovacao=10.0))
    assert regras.validar_desconto(5.0)
    assert not regras.validar_desconto(-1.0)
    assert not regras.validar_desconto(15.0)

    preco_com_desconto = regras.calcular_valor_com_desconto(500_000.0, 10.0)
    assert preco_com_desconto == 450_000.0


def test_pyfunc_rejeita_desconto_invalido() -> None:
    """Valida que o wrapper PyFunc rejeita descontos além do teto configurado."""
    pipeline = Pipeline(steps=[("modelo", LinearRegression())])
    pyfunc = ModeloImobiliarioPyFunc(
        modelo_pipeline=pipeline,
        desconto_maximo_permitido=5.0,
    )
    df = pd.DataFrame({"Metragem": [100.0], "Percentual_Desconto": [12.0]})
    with pytest.raises(ValueError, match="superior ao teto configurado"):
        pyfunc.predict(context=None, model_input=df)
