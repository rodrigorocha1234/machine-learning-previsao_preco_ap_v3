"""Calculador de pesos para voting inversamente proporcionais ao RMSE."""

import numpy as np


class CalculadorPesosRmse:
    """Calcula pesos normalizados w_i = (1/RMSE_i) / sum(1/RMSE_j)."""

    def calcular(self, rmse_por_modelo: dict[str, float]) -> dict[str, float]:
        """Calcula os pesos com base no erro validado.

        Parameters
        ----------
        rmse_por_modelo : dict[str, float]
            Mapeamento de nome do modelo para seu RMSE médio.

        Returns
        -------
        dict[str, float]
            Pesos normalizados somando 1.0.
        """
        if not rmse_por_modelo:
            return {}

        modelos = list(rmse_por_modelo.keys())
        inversos = np.array([1.0 / max(rmse_por_modelo[m], 1e-6) for m in modelos])
        soma = float(np.sum(inversos))

        pesos_norm = inversos / soma
        return {modelos[i]: float(pesos_norm[i]) for i in range(len(modelos))}
