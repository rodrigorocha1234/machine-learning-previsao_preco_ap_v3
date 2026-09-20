"""Analisador descritivo com todas as métricas estatísticas obrigatórias da EDA."""

import numpy as np
import pandas as pd
from scipy import stats


class AnalisadorDescritivo:
    """Calculador de estatísticas descritivas univariadas para dados tabulares."""

    PERCENTIS: tuple[float, ...] = (1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0)

    def analisar_numericas(
        self,
        dados: pd.DataFrame,
        colunas_numericas: list[str],
    ) -> dict[str, dict[str, float | int]]:
        """Calcula todas as 20+ estatísticas descritivas para colunas numéricas.

        Parameters
        ----------
        dados : pd.DataFrame
            DataFrame de entrada.
        colunas_numericas : list[str]
            Lista de nomes das colunas numéricas.

        Returns
        -------
        dict[str, dict[str, float | int]]
            Mapeamento de coluna para suas estatísticas calculadas.
        """
        resultado: dict[str, dict[str, float | int]] = {}

        for col in colunas_numericas:
            if col not in dados.columns:
                continue

            serie = dados[col].dropna()
            total_linhas = len(dados[col])
            ausentes = int(dados[col].isna().sum())

            if len(serie) == 0:
                continue

            media = float(serie.mean())
            mediana = float(serie.median())
            moda_val = float(serie.mode().iloc[0]) if not serie.mode().empty else float("nan")
            minimo = float(serie.min())
            maximo = float(serie.max())
            amplitude = float(maximo - minimo)
            variancia = float(serie.var())
            desvio_padrao = float(serie.std())

            q1 = float(np.percentile(serie, 25))
            q3 = float(np.percentile(serie, 75))
            iqr = float(q3 - q1)

            assimetria = float(stats.skew(serie))
            curtose = float(stats.kurtosis(serie))
            cv = float(desvio_padrao / media) if media != 0 else float("nan")
            zeros = int((serie == 0).sum())
            unicos = int(serie.nunique())

            stats_col: dict[str, float | int] = {
                "quantidade": len(serie),
                "ausentes": ausentes,
                "taxa_ausentes": float(ausentes / total_linhas) if total_linhas > 0 else 0.0,
                "media": media,
                "mediana": mediana,
                "moda": moda_val,
                "minimo": minimo,
                "maximo": maximo,
                "amplitude": amplitude,
                "variancia": variancia,
                "desvio_padrao": desvio_padrao,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "assimetria": assimetria,
                "curtose": curtose,
                "coeficiente_variacao": cv,
                "zeros": zeros,
                "valores_unicos": unicos,
            }

            for p in self.PERCENTIS:
                stats_col[f"p{int(p)}"] = float(np.percentile(serie, p))

            resultado[col] = stats_col

        return resultado

    def analisar_categoricas(
        self,
        dados: pd.DataFrame,
        colunas_categoricas: list[str],
        coluna_alvo: str | None = None,
    ) -> dict[str, dict[str, float | int | str]]:
        """Calcula estatísticas descritivas para colunas categóricas.

        Parameters
        ----------
        dados : pd.DataFrame
            DataFrame de entrada.
        colunas_categoricas : list[str]
            Lista de nomes das colunas categóricas.
        coluna_alvo : str | None
            Nome da coluna alvo para cruzamento de métricas.

        Returns
        -------
        dict[str, dict[str, float | int | str]]
            Estatísticas descritivas das categorias.
        """
        resultado: dict[str, dict[str, float | int | str]] = {}

        for col in colunas_categoricas:
            if col not in dados.columns:
                continue

            serie = dados[col].astype(str)
            total = len(serie)
            ausentes = int(dados[col].isna().sum())
            contagens = serie.value_counts()
            cardinalidade = int(serie.nunique())
            moda_cat = str(contagens.index[0]) if not contagens.empty else ""
            freq_moda = int(contagens.iloc[0]) if not contagens.empty else 0
            freq_rel_moda = float(freq_moda / total) if total > 0 else 0.0

            # Categorias raras (< 5% de frequência)
            raras = contagens[contagens / total < 0.05].index.tolist()

            res_cat: dict[str, float | int | str] = {
                "cardinalidade": cardinalidade,
                "ausentes": ausentes,
                "moda": moda_cat,
                "frequencia_moda": freq_moda,
                "frequencia_relativa_moda": freq_rel_moda,
                "qtd_categorias_raras": len(raras),
                "categorias_raras": ", ".join(raras[:5]),
            }

            if coluna_alvo and coluna_alvo in dados.columns:
                media_alvo = dados.groupby(col)[coluna_alvo].mean()
                res_cat["media_alvo_por_moda"] = (
                    float(media_alvo.get(moda_cat, float("nan"))) if moda_cat in media_alvo else float("nan")
                )

            resultado[col] = res_cat

        return resultado
