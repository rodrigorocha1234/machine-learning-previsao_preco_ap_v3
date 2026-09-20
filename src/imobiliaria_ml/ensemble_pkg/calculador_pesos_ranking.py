"""Calculador de pesos para voting inversamente proporcionais ao ranking médio."""

import numpy as np


class CalculadorPesosRanking:
    """Calcula pesos normalizados w_i = (1/rank_i) / sum(1/rank_j)."""

    def calcular(self, ranking_por_modelo: dict[str, float]) -> dict[str, float]:
        """Calcula os pesos com base no ranking médio estatístico.

        Parameters
        ----------
        ranking_por_modelo : dict[str, float]
            Mapeamento de nome do modelo para seu ranking médio pós-Friedman.

        Returns
        -------
        dict[str, float]
            Pesos normalizados somando 1.0.
        """
        if not ranking_por_modelo:
            return {}

        modelos = list(ranking_por_modelo.keys())
        inversos = np.array([1.0 / max(ranking_por_modelo[m], 1e-6) for m in modelos])
        soma = float(np.sum(inversos))

        pesos_norm = inversos / soma
        return {modelos[i]: float(pesos_norm[i]) for i in range(len(modelos))}
