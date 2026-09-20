"""Detector de drift na distribuição das previsões do modelo."""

import numpy as np
from scipy.stats import ks_2samp
from .detector_drift_numerico import DetectorDriftNumerico


class DetectorDriftPredicao:
    """Monitora alterações estatísticas nas saídas previstas (Prediction Drift)."""

    def __init__(self) -> None:
        self._numerico = DetectorDriftNumerico()

    def calcular_drift_predicoes(
        self,
        predicoes_base: np.ndarray,
        predicoes_atuais: np.ndarray,
    ) -> dict[str, float]:
        """Compara a estabilidade entre o histórico de predições e as novas predições.

        Parameters
        ----------
        predicoes_base : np.ndarray
            Valores previstos no conjunto de referência.
        predicoes_atuais : np.ndarray
            Valores previstos nos novos dados de produção.

        Returns
        -------
        dict[str, float]
            Métricas de PSI, KS e variação média das previsões.
        """
        arr_base = np.asarray(predicoes_base, dtype=float)
        arr_atual = np.asarray(predicoes_atuais, dtype=float)

        psi = self._numerico.calcular_psi(arr_base, arr_atual)
        ks_res = ks_2samp(arr_base, arr_atual)

        media_base = float(np.mean(arr_base)) if len(arr_base) > 0 else 0.0
        media_atual = float(np.mean(arr_atual)) if len(arr_atual) > 0 else 0.0
        variacao_rel = (
            float(abs(media_atual - media_base) / max(abs(media_base), 1.0))
            if media_base != 0
            else 0.0
        )

        return {
            "psi": psi,
            "ks_estatistica": float(ks_res.statistic),
            "ks_p_valor": float(ks_res.pvalue),
            "variacao_media_relativa": variacao_rel,
        }
