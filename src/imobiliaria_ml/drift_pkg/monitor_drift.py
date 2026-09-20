"""Monitor central de Data Drift integrando numéricas, categóricas e predições."""

import numpy as np
import pandas as pd
from ..enums_pkg.nivel_drift import NivelDrift
from .detector_drift_numerico import DetectorDriftNumerico
from .detector_drift_categorico import DetectorDriftCategorico
from .detector_drift_predicao import DetectorDriftPredicao
from .resultado_drift import ResultadoDrift


class MonitorDrift:
    """Consolida as verificações de drift e classifica a severidade com NivelDrift."""

    def __init__(
        self,
        psi_limite_atencao: float = 0.10,
        psi_limite_forte: float = 0.25,
    ) -> None:
        self._psi_limite_atencao = psi_limite_atencao
        self._psi_limite_forte = psi_limite_forte
        self._numerico = DetectorDriftNumerico()
        self._categorico = DetectorDriftCategorico()
        self._predicao = DetectorDriftPredicao()

    def avaliar(
        self,
        df_base: pd.DataFrame,
        df_atual: pd.DataFrame,
        colunas_numericas: list[str],
        colunas_categoricas: list[str],
        predicoes_base: np.ndarray | None = None,
        predicoes_atuais: np.ndarray | None = None,
    ) -> ResultadoDrift:
        """Executa a verificação completa de drift entre o baseline e os dados recentes.

        Parameters
        ----------
        df_base : pd.DataFrame
            Conjunto de dados de referência (treino do modelo campeão).
        df_atual : pd.DataFrame
            Novos registros a monitorar.
        colunas_numericas : list[str]
            Lista de features contínuas.
        colunas_categoricas : list[str]
            Lista de features categóricas.
        predicoes_base : np.ndarray | None
            Previsões históricas correspondentes ao baseline.
        predicoes_atuais : np.ndarray | None
            Previsões geradas para o conjunto atual.

        Returns
        -------
        ResultadoDrift
            Diagnóstico consolidado tipado com NivelDrift.
        """
        res_num = self._numerico.detectar_drift(df_base, df_atual, colunas_numericas)
        res_cat = self._categorico.detectar_drift(df_base, df_atual, colunas_categoricas)

        psi_pred = 0.0
        if predicoes_base is not None and predicoes_atuais is not None:
            res_pred = self._predicao.calcular_drift_predicoes(predicoes_base, predicoes_atuais)
            psi_pred = float(res_pred.get("psi", 0.0))

        # Identifica o PSI máximo observado entre todas as features e a predição
        psis: list[float] = [psi_pred]
        for v in res_num.values():
            psis.append(float(v.get("psi", 0.0)))
        for v in res_cat.values():
            val = v.get("psi")
            if isinstance(val, (int, float)):
                psis.append(float(val))

        max_psi = max(psis) if psis else 0.0

        if max_psi >= self._psi_limite_forte:
            nivel = NivelDrift.FORTE
            detectado = True
        elif max_psi >= self._psi_limite_atencao:
            nivel = NivelDrift.ATENCAO
            detectado = True
        else:
            nivel = NivelDrift.ESTAVEL
            detectado = False

        detalhes: dict[str, str | float | int] = {
            "psi_maximo": max_psi,
            "total_linhas_base": len(df_base),
            "total_linhas_atual": len(df_atual),
            "nivel": str(nivel.value),
        }

        return ResultadoDrift(
            nivel_drift=nivel,
            drift_detectado=detectado,
            estatisticas_numericas=res_num,
            estatisticas_categoricas=res_cat,
            drift_predicao_psi=psi_pred,
            detalhes=detalhes,
        )
