"""Gerador do relatório consolidado e artefatos de EDA em memória."""

from matplotlib.figure import Figure
import pandas as pd
from .analisador_descritivo import AnalisadorDescritivo
from .analisador_distribuicao import AnalisadorDistribuicao
from .estatisticas_descritivas import EstatisticasDescritivas


class GeradorRelatorioEda:
    """Consolidador de dados, matrizes, métricas e figuras para a EDA."""

    def __init__(self) -> None:
        self._descritivo = AnalisadorDescritivo()
        self._distribuicao = AnalisadorDistribuicao()

    def gerar_relatorio_completo(
        self,
        dados: pd.DataFrame,
        colunas_numericas: list[str],
        colunas_categoricas: list[str],
        coluna_alvo: str = "Valor_da_Venda",
    ) -> tuple[
        EstatisticasDescritivas,
        pd.DataFrame,
        pd.DataFrame,
        dict[str, Figure],
    ]:
        """Gera todas as estatísticas, DataFrames tabulares e figuras em memória.

        Parameters
        ----------
        dados : pd.DataFrame
            Base de dados do projeto.
        colunas_numericas : list[str]
            Colunas numéricas a analisar.
        colunas_categoricas : list[str]
            Colunas categóricas a analisar.
        coluna_alvo : str
            Coluna alvo contínua.

        Returns
        -------
        tuple[EstatisticasDescritivas, pd.DataFrame, pd.DataFrame, dict[str, Figure]]
            Estatísticas consolidadas, DataFrame descritivo, DataFrame de categorias e dicionário de figuras.
        """
        # Análise univariada
        stats_num = self._descritivo.analisar_numericas(dados, colunas_numericas)
        stats_cat = self._descritivo.analisar_categoricas(dados, colunas_categoricas, coluna_alvo)

        # Relações e multicolinearidade
        features_num = [c for c in colunas_numericas if c != coluna_alvo]
        vif_dict = self._distribuicao.calcular_vif(dados, features_num)
        outliers_dict = self._distribuicao.diagnosticar_outliers_iqr(dados, colunas_numericas)

        estatisticas = EstatisticasDescritivas(
            tabela_numerica=stats_num,
            tabela_categorica=stats_cat,
            tabela_vif=vif_dict,
            outliers_iqr=outliers_dict,
        )

        df_descritivo = pd.DataFrame(stats_num).T
        df_categorias = pd.DataFrame(stats_cat).T

        # Geração de Figuras em memória
        matriz_pearson = self._distribuicao.calcular_correlacao_pearson(dados, colunas_numericas)
        matriz_spearman = self._distribuicao.calcular_correlacao_spearman(dados, colunas_numericas)

        figuras: dict[str, Figure] = {
            "correlacao_pearson": self._distribuicao.gerar_figura_correlacao(
                matriz_pearson, "Matriz de Correlação de Pearson"
            ),
            "correlacao_spearman": self._distribuicao.gerar_figura_correlacao(
                matriz_spearman, "Matriz de Correlação de Spearman"
            ),
            "distribuicao_alvo": self._distribuicao.gerar_figura_distribuicao_alvo(
                dados, coluna_alvo
            ),
        }

        if "Zona" in dados.columns and coluna_alvo in dados.columns:
            figuras["boxplot_zona_valor"] = self._distribuicao.gerar_figura_boxplot_zona(
                dados, "Zona", coluna_alvo
            )

        return estatisticas, df_descritivo, df_categorias, figuras
