"""Analisador de distribuições, correlações, VIF e figuras da EDA."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


class AnalisadorDistribuicao:
    """Calculador de correlações bivariadas, VIF e figuras gráficas."""

    def calcular_correlacao_pearson(
        self,
        dados: pd.DataFrame,
        colunas_numericas: list[str],
    ) -> pd.DataFrame:
        """Calcula matriz de correlação linear de Pearson."""
        validas = [c for c in colunas_numericas if c in dados.columns]
        return dados[validas].corr(method="pearson")

    def calcular_correlacao_spearman(
        self,
        dados: pd.DataFrame,
        colunas_numericas: list[str],
    ) -> pd.DataFrame:
        """Calcula matriz de correlação monotônica de Spearman."""
        validas = [c for c in colunas_numericas if c in dados.columns]
        return dados[validas].corr(method="spearman")

    def calcular_vif(
        self,
        dados: pd.DataFrame,
        colunas_features: list[str],
    ) -> dict[str, float]:
        """Calcula o Fator de Inflação da Variância (VIF) para verificar multicolinearidade.

        Parameters
        ----------
        dados : pd.DataFrame
            DataFrame com as variáveis.
        colunas_features : list[str]
            Colunas numéricas preditoras (sem o alvo).

        Returns
        -------
        dict[str, float]
            VIF de cada variável preditora.
        """
        validas = [c for c in colunas_features if c in dados.columns]
        if len(validas) < 2:
            return {}

        df_limpo = dados[validas].dropna()
        if len(df_limpo) < len(validas) + 1:
            return {}

        # Adiciona constante para regressão do VIF
        valores = df_limpo.values
        vif_dict: dict[str, float] = {}

        for i, col in enumerate(validas):
            try:
                vif = float(variance_inflation_factor(valores, i))
                vif_dict[col] = vif
            except Exception:
                vif_dict[col] = float("nan")

        return vif_dict

    def diagnosticar_outliers_iqr(
        self,
        dados: pd.DataFrame,
        colunas_numericas: list[str],
    ) -> dict[str, int]:
        """Identifica quantidade de outliers univariados pela regra do IQR."""
        outliers: dict[str, int] = {}
        for col in colunas_numericas:
            if col not in dados.columns:
                continue
            serie = dados[col].dropna()
            q1 = float(np.percentile(serie, 25))
            q3 = float(np.percentile(serie, 75))
            iqr = q3 - q1
            limite_inf = q1 - 1.5 * iqr
            limite_sup = q3 + 1.5 * iqr
            qtd = int(((serie < limite_inf) | (serie > limite_sup)).sum())
            outliers[col] = qtd
        return outliers

    def gerar_figura_correlacao(
        self,
        matriz: pd.DataFrame,
        titulo: str,
    ) -> Figure:
        """Gera em memória uma figura de mapa de calor da correlação."""
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(matriz.values, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(cax)

        ax.set_xticks(range(len(matriz.columns)))
        ax.set_yticks(range(len(matriz.index)))
        ax.set_xticklabels(matriz.columns, rotation=45, ha="left")
        ax.set_yticklabels(matriz.index)
        ax.set_title(titulo, pad=20)

        # Adiciona valores textuais
        for i in range(len(matriz.index)):
            for j in range(len(matriz.columns)):
                val = matriz.iloc[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(val) > 0.5 else "black",
                )

        fig.tight_layout()
        return fig

    def gerar_figura_distribuicao_alvo(
        self,
        dados: pd.DataFrame,
        coluna_alvo: str = "Valor_da_Venda",
    ) -> Figure:
        """Gera em memória um histograma com curva de densidade do alvo."""
        fig, ax = plt.subplots(figsize=(8, 5))
        valores = dados[coluna_alvo].dropna()

        ax.hist(valores, bins=40, density=True, alpha=0.6, color="#1f77b4", edgecolor="black")
        ax.set_title(f"Distribuição do Alvo: {coluna_alvo}")
        ax.set_xlabel("Valor (R$)")
        ax.set_ylabel("Densidade")
        fig.tight_layout()
        return fig

    def gerar_figura_boxplot_zona(
        self,
        dados: pd.DataFrame,
        coluna_categoria: str = "Zona",
        coluna_alvo: str = "Valor_da_Venda",
    ) -> Figure:
        """Gera em memória boxplots agrupados por Zona imobiliária."""
        fig, ax = plt.subplots(figsize=(10, 6))

        grupos: list[tuple[str, np.ndarray]] = []
        for cat in dados[coluna_categoria].dropna().unique():
            sub = dados[dados[coluna_categoria] == cat][coluna_alvo].dropna().values
            if len(sub) > 0:
                grupos.append((str(cat), sub))

        rotulos = [g[0] for g in grupos]
        dados_boxplot = [g[1] for g in grupos]

        if dados_boxplot:
            ax.boxplot(dados_boxplot, tick_labels=rotulos)
            ax.set_xticklabels(rotulos, rotation=30, ha="right")

        ax.set_title(f"Distribuição de {coluna_alvo} por {coluna_categoria}")
        ax.set_xlabel(coluna_categoria)
        ax.set_ylabel("Valor (R$)")
        fig.tight_layout()
        return fig
