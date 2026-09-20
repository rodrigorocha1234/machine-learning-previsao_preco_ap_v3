"""Detector de drift para atributos numéricos com Kolmogorov-Smirnov e PSI."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


class DetectorDriftNumerico:
    """Calcula estatísticas KS e PSI entre dados de referência e dados novos."""

    def calcular_psi(
        self,
        base: np.ndarray,
        atual: np.ndarray,
        num_bins: int = 10,
    ) -> float:
        """Calcula o Population Stability Index (PSI) de uma série contínua."""
        if len(base) == 0 or len(atual) == 0:
            return 0.0

        quantis = np.linspace(0, 100, num_bins + 1)
        limites = np.percentile(base, quantis)
        limites[0] = -float("inf")
        limites[-1] = float("inf")
        limites = np.unique(limites)

        cont_base, _ = np.histogram(base, bins=limites)
        cont_atual, _ = np.histogram(atual, bins=limites)

        prop_base = cont_base / len(base)
        prop_atual = cont_atual / len(atual)

        # Suavização para evitar log(0) e divisão por 0
        prop_base = np.where(prop_base == 0, 1e-4, prop_base)
        prop_atual = np.where(prop_atual == 0, 1e-4, prop_atual)

        psi = np.sum((prop_atual - prop_base) * np.log(prop_atual / prop_base))
        return float(psi)

    def detectar_drift(
        self,
        df_base: pd.DataFrame,
        df_atual: pd.DataFrame,
        colunas: list[str],
    ) -> dict[str, dict[str, float]]:
        """Aplica teste KS e PSI em cada coluna numérica presente em ambos os conjuntos."""
        resultados: dict[str, dict[str, float]] = {}

        for col in colunas:
            if col not in df_base.columns or col not in df_atual.columns:
                continue

            vals_base = df_base[col].dropna().values
            vals_atual = df_atual[col].dropna().values

            if len(vals_base) == 0 or len(vals_atual) == 0:
                continue

            ks_res = ks_2samp(vals_base, vals_atual)
            psi_val = self.calcular_psi(vals_base, vals_atual)

            media_base = float(np.mean(vals_base))
            media_atual = float(np.mean(vals_atual))
            dif_media_rel = float(
                abs(media_atual - media_base) / max(abs(media_base), 1e-6)
            )

            resultados[col] = {
                "ks_estatistica": float(ks_res.statistic),
                "ks_p_valor": float(ks_res.pvalue),
                "psi": psi_val,
                "diferenca_media_relativa": dif_media_rel,
            }

        return resultados
