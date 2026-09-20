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
        """Gera boxplots enriquecidos de preço agrupados e ordenados por Zona imobiliária."""
        fig, ax = plt.subplots(figsize=(10, 6))

        if coluna_categoria not in dados.columns or coluna_alvo not in dados.columns:
            return fig

        grupos: list[tuple[str, np.ndarray, float]] = []
        for cat in dados[coluna_categoria].dropna().unique():
            sub = dados[dados[coluna_categoria] == cat][coluna_alvo].dropna().values
            if len(sub) > 0:
                mediana = float(np.median(sub))
                grupos.append((str(cat), sub, mediana))

        # Ordena as zonas por mediana decrescente para interpretação mercadológica direta
        grupos.sort(key=lambda g: g[2], reverse=True)

        rotulos = [f"{g[0]}\n(n={len(g[1])})" for g in grupos]
        dados_boxplot = [g[1] for g in grupos]

        if dados_boxplot:
            paleta_cores = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]
            bp = ax.boxplot(
                dados_boxplot,
                tick_labels=rotulos,
                patch_artist=True,
                medianprops=dict(color="#d62728", linewidth=2.2),
                flierprops=dict(marker="o", markersize=4, alpha=0.5, markerfacecolor="#7f7f7f"),
            )
            for i, box in enumerate(bp["boxes"]):
                cor = paleta_cores[i % len(paleta_cores)]
                box.set_facecolor(cor)
                box.set_alpha(0.65)
                box.set_edgecolor("#333333")

            # Anotações dos valores das medianas acima de cada caixa
            for i, g in enumerate(grupos):
                mediana_val = g[2]
                texto_mediana = f"R$ {mediana_val / 1000:,.0f}k" if mediana_val >= 1000 else f"R$ {mediana_val:,.0f}"
                ax.text(
                    i + 1,
                    mediana_val,
                    f" {texto_mediana}",
                    va="center",
                    ha="left",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#800000",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", alpha=0.8, edgecolor="#d62728", linewidth=0.8),
                )

        ax.set_title(f"Distribuição de Preço ({coluna_alvo}) por {coluna_categoria} (Ordenado por Mediana)", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"{coluna_categoria} (com volume amostral)", fontsize=10)
        ax.set_ylabel("Valor de Venda (R$)", fontsize=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"R$ {x:,.0f}"))
        fig.tight_layout()
        return fig

    def gerar_figura_boxplot_zona_m2(
        self,
        dados: pd.DataFrame,
        coluna_categoria: str = "Zona",
        coluna_alvo: str = "Valor_da_Venda",
        coluna_metragem: str = "Metragem",
    ) -> Figure:
        """Gera boxplots de Preço por Metro Quadrado (R$/m²) agrupados por Zona."""
        fig, ax = plt.subplots(figsize=(10, 6))

        if not {coluna_categoria, coluna_alvo, coluna_metragem}.issubset(dados.columns):
            return fig

        df_m2 = dados.dropna(subset=[coluna_categoria, coluna_alvo, coluna_metragem]).copy()
        metragens = np.maximum(df_m2[coluna_metragem].values, 1.0)
        df_m2["Valor_m2"] = df_m2[coluna_alvo].values / metragens

        grupos: list[tuple[str, np.ndarray, float]] = []
        for cat in df_m2[coluna_categoria].unique():
            sub = df_m2[df_m2[coluna_categoria] == cat]["Valor_m2"].values
            if len(sub) > 0:
                grupos.append((str(cat), sub, float(np.median(sub))))

        grupos.sort(key=lambda g: g[2], reverse=True)
        rotulos = [f"{g[0]}\n(n={len(g[1])})" for g in grupos]
        dados_boxplot = [g[1] for g in grupos]

        if dados_boxplot:
            paleta = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#1abc9c", "#e74c3c"]
            bp = ax.boxplot(
                dados_boxplot,
                tick_labels=rotulos,
                patch_artist=True,
                medianprops=dict(color="#c0392b", linewidth=2.2),
                flierprops=dict(marker="o", markersize=4, alpha=0.4),
            )
            for i, box in enumerate(bp["boxes"]):
                box.set_facecolor(paleta[i % len(paleta)])
                box.set_alpha(0.65)

            for i, g in enumerate(grupos):
                med = g[2]
                ax.text(
                    i + 1,
                    med,
                    f" R$ {med:,.0f}/m²",
                    va="center",
                    ha="left",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#78281f",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", alpha=0.85, edgecolor="#c0392b", linewidth=0.8),
                )

        ax.set_title("Preço Médio por Metro Quadrado (R$/m²) por Zona", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"{coluna_categoria} (ordenado por valor do m²)", fontsize=10)
        ax.set_ylabel("Valor por m² (R$/m²)", fontsize=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"R$ {x:,.0f}"))
        fig.tight_layout()
        return fig

    def gerar_figura_perfil_zonas(
        self,
        dados: pd.DataFrame,
        coluna_categoria: str = "Zona",
        coluna_alvo: str = "Valor_da_Venda",
        coluna_metragem: str = "Metragem",
    ) -> Figure:
        """Gera painel visual com volume e ticket médio de mercado por Zona imobiliária."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        if coluna_categoria not in dados.columns or coluna_alvo not in dados.columns:
            return fig

        agrupado = dados.groupby(coluna_categoria, observed=False).agg(
            quantidade=(coluna_alvo, "count"),
            preco_medio=(coluna_alvo, "mean"),
            preco_mediano=(coluna_alvo, "median"),
        ).reset_index()

        if coluna_metragem in dados.columns:
            m2_series = dados[coluna_alvo] / np.maximum(dados[coluna_metragem], 1.0)
            agrupado["m2_medio"] = dados.groupby(coluna_categoria, observed=False).apply(
                lambda g: (g[coluna_alvo] / np.maximum(g[coluna_metragem], 1.0)).mean()
            ).values

        agrupado = agrupado.sort_values(by="preco_medio", ascending=True).reset_index(drop=True)
        zonas = agrupado[coluna_categoria].astype(str).values

        # Gráfico 1: Preço Médio e Mediano
        y_pos = np.arange(len(zonas))
        ax1.barh(y_pos - 0.2, agrupado["preco_medio"], height=0.4, label="Média", color="#2b5c8f", alpha=0.85)
        ax1.barh(y_pos + 0.2, agrupado["preco_mediano"], height=0.4, label="Mediana", color="#48c9b0", alpha=0.85)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(zonas, fontsize=9.5)
        ax1.set_xlabel("Valor de Venda (R$)", fontsize=10)
        ax1.set_title("Ticket Médio vs Mediano por Zona", fontsize=11, fontweight="bold")
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"R$ {x/1000:,.0f}k"))
        ax1.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(loc="lower right")

        # Gráfico 2: Participação amostral (%)
        total = agrupado["quantidade"].sum()
        pcts = (agrupado["quantidade"] / max(total, 1)) * 100.0
        barras = ax2.barh(y_pos, pcts, color="#f39c12", alpha=0.85, height=0.55)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(zonas, fontsize=9.5)
        ax2.set_xlabel("Proporção de Imóveis no Dataset (%)", fontsize=10)
        ax2.set_title("Participação Amostral por Zona", fontsize=11, fontweight="bold")
        ax2.xaxis.grid(True, linestyle="--", alpha=0.4)
        for barra in barras:
            largura = barra.get_width()
            ax2.text(largura + 0.8, barra.get_y() + barra.get_height() / 2, f"{largura:.1f}%", va="center", fontsize=9)

        fig.tight_layout()
        return fig

