"""Testes unitários para componentes de Análise Exploratória de Dados (EDA)."""

from matplotlib.figure import Figure
import pandas as pd
from imobiliaria_ml.eda_pkg.analisador_descritivo import AnalisadorDescritivo
from imobiliaria_ml.eda_pkg.analisador_distribuicao import AnalisadorDistribuicao
from imobiliaria_ml.eda_pkg.gerador_relatorio_eda import GeradorRelatorioEda


def test_analisador_descritivo_calcula_estatisticas(df_imoveis_sintetico: pd.DataFrame) -> None:
    analisador = AnalisadorDescritivo()
    res = analisador.analisar_numericas(df_imoveis_sintetico, ["Quartos", "Metragem"])
    assert "Metragem" in res
    assert "media" in res["Metragem"]
    assert "mediana" in res["Metragem"]
    assert "q1" in res["Metragem"]
    assert "q3" in res["Metragem"]
    assert "assimetria" in res["Metragem"]


def test_analisador_distribuicao_gera_figuras(df_imoveis_sintetico: pd.DataFrame) -> None:
    dist = AnalisadorDistribuicao()
    matriz_pearson = dist.calcular_correlacao_pearson(df_imoveis_sintetico, ["Quartos", "Metragem"])
    assert matriz_pearson.shape == (2, 2)

    fig = dist.gerar_figura_correlacao(matriz_pearson, "Teste")
    assert isinstance(fig, Figure)


def test_gerador_relatorio_eda(df_imoveis_sintetico: pd.DataFrame) -> None:
    gerador = GeradorRelatorioEda()
    stats, df_num, df_cat, figs = gerador.gerar_relatorio_completo(
        dados=df_imoveis_sintetico,
        colunas_numericas=["Quartos", "Banheiros", "Metragem", "Valor_da_Venda"],
        colunas_categoricas=["Zona"],
    )
    assert not df_num.empty
    assert not df_cat.empty
    assert "correlacao_pearson" in figs
    assert isinstance(figs["correlacao_pearson"], Figure)
