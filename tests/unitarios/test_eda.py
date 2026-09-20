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
    assert "boxplot_zona_valor" in figs
    assert "boxplot_zona_m2" in figs
    assert "perfil_imobiliario_zona" in figs
    assert not stats.tabela_zonas_df.empty
    assert "zona_mais_valorizada" in stats.diagnostico_zona
    assert len(stats.resumo_zonas_md) > 50


def test_interpretador_eda_analise_por_zona(df_imoveis_sintetico: pd.DataFrame) -> None:
    from imobiliaria_ml.eda_pkg.interpretador_eda import InterpretadorEda

    interpretador = InterpretadorEda()
    df_zonas = interpretador.analisar_estatisticas_por_zona(
        dados=df_imoveis_sintetico,
        coluna_zona="Zona",
        coluna_alvo="Valor_da_Venda",
        coluna_metragem="Metragem",
    )

    assert not df_zonas.empty
    assert "zona" in df_zonas.columns
    assert "preco_medio" in df_zonas.columns
    assert "preco_m2_medio" in df_zonas.columns
    assert "coeficiente_variacao_pct" in df_zonas.columns

    diagnostico = interpretador.gerar_diagnostico_zonas(df_zonas)
    assert "zona_mais_valorizada" in diagnostico
    assert "zona_mais_acessivel" in diagnostico
    assert "fator_gradiente_espacial" in diagnostico
    assert "implicacao_modelagem" in diagnostico

    texto_md = interpretador.gerar_resumo_textual(df_zonas, diagnostico)
    assert "# Interpretação da Análise Exploratória de Dados por Zona" in texto_md
    assert "Tabela Comparativa de Indicadores" in texto_md


def test_analisador_distribuicao_figuras_zona(df_imoveis_sintetico: pd.DataFrame) -> None:
    dist = AnalisadorDistribuicao()

    fig_box = dist.gerar_figura_boxplot_zona(df_imoveis_sintetico, "Zona", "Valor_da_Venda")
    assert isinstance(fig_box, Figure)

    fig_m2 = dist.gerar_figura_boxplot_zona_m2(df_imoveis_sintetico, "Zona", "Valor_da_Venda", "Metragem")
    assert isinstance(fig_m2, Figure)

    fig_perfil = dist.gerar_figura_perfil_zonas(df_imoveis_sintetico, "Zona", "Valor_da_Venda", "Metragem")
    assert isinstance(fig_perfil, Figure)

