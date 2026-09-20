"""Gerador do relatório consolidado e artefatos de EDA em memória."""

from matplotlib.figure import Figure
import pandas as pd
from .analisador_descritivo import AnalisadorDescritivo
from .analisador_distribuicao import AnalisadorDistribuicao
from .estatisticas_descritivas import EstatisticasDescritivas
from .interpretador_eda import InterpretadorEda


class GeradorRelatorioEda:
    """Consolidador de dados, matrizes, métricas e figuras para a EDA."""

    def __init__(self) -> None:
        self._descritivo = AnalisadorDescritivo()
        self._distribuicao = AnalisadorDistribuicao()
        self._interpretador = InterpretadorEda()

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

        # Análise e interpretação aprofundada por Zona
        df_zonas = pd.DataFrame()
        diag_zonas: dict[str, str] = {}
        resumo_zonas_md = ""
        relatorio_negocio_md = ""
        analise_zona_dict: dict[str, dict[str, float | int | str]] = {}

        if "Zona" in dados.columns and coluna_alvo in dados.columns:
            df_zonas = self._interpretador.analisar_estatisticas_por_zona(
                dados=dados,
                coluna_zona="Zona",
                coluna_alvo=coluna_alvo,
                coluna_metragem="Metragem",
            )
            diag_zonas = self._interpretador.gerar_diagnostico_zonas(df_zonas)
            resumo_zonas_md = self._interpretador.gerar_resumo_textual(df_zonas, diag_zonas)
            relatorio_negocio_md = self._interpretador.gerar_relatorio_negocio(
                tabela_zonas=df_zonas,
                diagnostico=diag_zonas,
                stats_numericas=stats_num,
                outliers_iqr=outliers_dict,
                coluna_alvo=coluna_alvo,
            )
            if not df_zonas.empty and "zona" in df_zonas.columns:
                for _, r in df_zonas.iterrows():
                    z_nome = str(r["zona"])
                    analise_zona_dict[z_nome] = {k: v for k, v in r.items() if k != "zona"}

        estatisticas = EstatisticasDescritivas(
            tabela_numerica=stats_num,
            tabela_categorica=stats_cat,
            tabela_vif=vif_dict,
            outliers_iqr=outliers_dict,
            analise_zona=analise_zona_dict,
            diagnostico_zona=diag_zonas,
            tabela_zonas_df=df_zonas,
            resumo_zonas_md=resumo_zonas_md,
            relatorio_negocio_md=relatorio_negocio_md,
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
            if "Metragem" in dados.columns:
                figuras["boxplot_zona_m2"] = self._distribuicao.gerar_figura_boxplot_zona_m2(
                    dados, "Zona", coluna_alvo, "Metragem"
                )
            figuras["perfil_imobiliario_zona"] = self._distribuicao.gerar_figura_perfil_zonas(
                dados, "Zona", coluna_alvo, "Metragem" if "Metragem" in dados.columns else ""
            )

        return estatisticas, df_descritivo, df_categorias, figuras
