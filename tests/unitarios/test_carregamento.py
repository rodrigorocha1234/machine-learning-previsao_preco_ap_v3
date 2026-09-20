"""Testes unitários para carregadores de dados e factory."""

from pathlib import Path
import pandas as pd
import pytest
from imobiliaria_ml.enums_pkg.tipo_carregador import TipoCarregador
from imobiliaria_ml.carregamento_pkg.fabrica_carregadores import FabricaCarregadores
from imobiliaria_ml.carregamento_pkg.carregador_csv import CarregadorCsv
from imobiliaria_ml.carregamento_pkg.carregador_excel import CarregadorExcel


def test_fabrica_cria_carregador_csv() -> None:
    fabrica = FabricaCarregadores()
    carregador = fabrica.criar(TipoCarregador.CSV)
    assert isinstance(carregador, CarregadorCsv)


def test_fabrica_cria_carregador_excel() -> None:
    fabrica = FabricaCarregadores()
    carregador = fabrica.criar(TipoCarregador.EXCEL)
    assert isinstance(carregador, CarregadorExcel)


def test_carregador_csv_arquivo_inexistente() -> None:
    carregador = CarregadorCsv()
    with pytest.raises(FileNotFoundError):
        carregador.carregar_dados("caminho_inexistente.csv")


def test_carregador_csv_carrega_dados_reais(tmp_path: Path) -> None:
    arquivo = tmp_path / "teste.csv"
    arquivo.write_text("col1,col2\n1,2\n3,4")
    carregador = CarregadorCsv()
    df = carregador.carregar_dados(arquivo)
    assert len(df) == 2
    assert list(df.columns) == ["col1", "col2"]
