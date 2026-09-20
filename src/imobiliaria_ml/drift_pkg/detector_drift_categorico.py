"""Detector de drift para atributos categóricos com Qui-Quadrado e PSI categórico."""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


class DetectorDriftCategorico:
    """Mede mudanças na distribuição e estabilidade de variáveis categóricas."""

    def detectar_drift(
        self,
        df_base: pd.DataFrame,
        df_atual: pd.DataFrame,
        colunas: list[str],
    ) -> dict[str, dict[str, float | int | str]]:
        """Avalia drift de categorias entre a base de referência e novos registros."""
        resultados: dict[str, dict[str, float | int | str]] = {}

        for col in colunas:
            if col not in df_base.columns or col not in df_atual.columns:
                continue

            s_base = df_base[col].astype(str)
            s_atual = df_atual[col].astype(str)

            cats_base = set(s_base.unique())
            cats_atual = set(s_atual.unique())

            novas_cats = sorted(list(cats_atual - cats_base))
            cats_desaparecidas = sorted(list(cats_base - cats_atual))

            todas_cats = sorted(list(cats_base.union(cats_atual)))

            cont_base = [int((s_base == c).sum()) for c in todas_cats]
            cont_atual = [int((s_atual == c).sum()) for c in todas_cats]

            # PSI Categórico
            tot_b = max(len(s_base), 1)
            tot_a = max(len(s_atual), 1)

            prop_b = np.array([max(c / tot_b, 1e-4) for c in cont_base])
            prop_a = np.array([max(c / tot_a, 1e-4) for c in cont_atual])

            psi_cat = float(np.sum((prop_a - prop_b) * np.log(prop_a / prop_b)))

            # Teste Qui-Quadrado de independência
            tabela = np.array([cont_base, cont_atual])
            # Evita erro em contingency table se alguma linha for totalmente nula
            if (tabela > 0).any(axis=1).all():
                chi2_stat, p_val, _, _ = chi2_contingency(tabela)
            else:
                chi2_stat, p_val = 0.0, 1.0

            resultados[col] = {
                "psi": psi_cat,
                "qui_quadrado": float(chi2_stat),
                "p_valor": float(p_val),
                "novas_categorias_qtd": len(novas_cats),
                "novas_categorias": ", ".join(novas_cats[:5]),
                "desaparecidas_qtd": len(cats_desaparecidas),
            }

        return resultados
