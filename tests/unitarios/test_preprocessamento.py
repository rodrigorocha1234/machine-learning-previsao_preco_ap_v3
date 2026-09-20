"""Testes unitários para pré-processamento e escalonamento."""

import pandas as pd
from imobiliaria_ml.enums_pkg.tipo_escalonador import TipoEscalonador
from imobiliaria_ml.preprocessamento_pkg.fabrica_escalonadores import FabricaEscalonadores
from imobiliaria_ml.preprocessamento_pkg.escalonador_standard import EscalonadorStandard
from imobiliaria_ml.preprocessamento_pkg.pre_processador import PreProcessador


def test_fabrica_escalonadores_instancia_corretamente() -> None:
    fabrica = FabricaEscalonadores()
    escalonador = fabrica.criar(TipoEscalonador.STANDARD)
    assert isinstance(escalonador, EscalonadorStandard)


def test_pre_processador_transforma_dados(df_imoveis_sintetico: pd.DataFrame) -> None:
    pre = PreProcessador(estrategia_escalonamento=EscalonadorStandard())
    ct = pre.construir_pipeline(
        colunas_numericas=["Quartos", "Banheiros", "Vagas", "Metragem"],
        colunas_categoricas=["Zona"],
    )
    X = df_imoveis_sintetico.drop(columns=["Valor_da_Venda"])
    X_trans = ct.fit_transform(X)
    assert X_trans.shape[0] == len(df_imoveis_sintetico)
    assert X_trans.shape[1] >= 5
