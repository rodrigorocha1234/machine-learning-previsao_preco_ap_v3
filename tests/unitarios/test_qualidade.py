"""Testes unitários para validadores de esquema e qualidade."""

import pandas as pd
from imobiliaria_ml.qualidade_pkg.validador_esquema import ValidadorEsquema
from imobiliaria_ml.qualidade_pkg.validador_qualidade import ValidadorQualidade


def test_validador_esquema_sucesso(df_imoveis_sintetico: pd.DataFrame) -> None:
    validador = ValidadorEsquema()
    res = validador.validar(df_imoveis_sintetico, exigir_alvo=True)
    assert res.valido
    assert len(res.mensagens_erro) == 0


def test_validador_qualidade_detecta_duplicatas() -> None:
    validador = ValidadorQualidade()
    df = pd.DataFrame(
        {
            "Zona": ["Centro", "Centro"],
            "Quartos": [2, 2],
            "Banheiros": [1, 1],
            "Vagas": [1, 1],
            "Metragem": [80.0, 80.0],
            "Valor_da_Venda": [400_000.0, 400_000.0],
        }
    )
    res = validador.validar(df)
    assert res.valido
    assert any("Detectadas 1 linhas duplicadas" in a for a in res.mensagens_aviso)


def test_validador_qualidade_rejeita_nulos_excessivos() -> None:
    validador = ValidadorQualidade(limite_taxa_nulos=0.30)
    df = pd.DataFrame(
        {
            "Zona": [None, None, "Centro"],
            "Quartos": [2, 3, 4],
        }
    )
    res = validador.validar(df)
    assert not res.valido
    assert any("ultrapassou o limite de nulos" in err for err in res.mensagens_erro)
