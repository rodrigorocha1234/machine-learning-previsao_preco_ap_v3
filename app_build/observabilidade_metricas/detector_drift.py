from dataclasses import dataclass

import numpy as np
import scipy.stats


@dataclass(frozen=True)
class ResultadoDrift:
    psi: float
    ks_estatistica: float
    ks_pvalor: float
    wasserstein_distancia: float
    status_severidade: int


class DetectorDrift:
    def __init__(self, numero_baldes: int = 10) -> None:
        self._numero_baldes = numero_baldes

    def calcular_psi(
        self,
        base_referencia: np.ndarray,
        base_producao: np.ndarray,
    ) -> float:
        percentis = np.linspace(0, 100, self._numero_baldes + 1)
        limites = np.percentile(base_referencia, percentis)
        limites_unicos = np.unique(limites)

        tamanho_suficiente = len(limites_unicos) > 1
        limites_finais = {
            True: limites_unicos,
            False: np.array([float(np.min(base_referencia)) - 1.0, float(np.max(base_referencia)) + 1.0]),
        }[tamanho_suficiente]

        contagens_ref, _ = np.histogram(base_referencia, bins=limites_finais)
        contagens_prod, _ = np.histogram(base_producao, bins=limites_finais)

        qtd_ref = float(len(base_referencia)) + float(len(contagens_ref))
        qtd_prod = float(len(base_producao)) + float(len(contagens_prod))

        proporcao_ref = np.maximum(contagens_ref, 1) / qtd_ref
        proporcao_prod = np.maximum(contagens_prod, 1) / qtd_prod

        diferenca = proporcao_prod - proporcao_ref
        razao = np.log(proporcao_prod / proporcao_ref)
        psi_vetorial = diferenca * razao

        return float(np.sum(psi_vetorial))

    def calcular_ks(
        self,
        base_referencia: np.ndarray,
        base_producao: np.ndarray,
    ) -> tuple[float, float]:
        resultado = scipy.stats.ks_2samp(base_referencia, base_producao)
        return float(resultado.statistic), float(resultado.pvalue)

    def calcular_wasserstein(
        self,
        base_referencia: np.ndarray,
        base_producao: np.ndarray,
    ) -> float:
        distancia = scipy.stats.wasserstein_distance(base_referencia, base_producao)
        desvio_padrao = float(np.std(base_referencia)) + 1e-6
        return float(distancia / desvio_padrao)

    def classificar_severidade(self, psi: float, ks_pvalor: float) -> int:
        critico = bool((psi >= 0.20) or (ks_pvalor < 0.01))
        moderado = bool((psi >= 0.10) or (ks_pvalor < 0.05))

        tabela_severidade: dict[tuple[bool, bool], int] = {
            (True, True): 2,
            (False, True): 1,
            (False, False): 0,
            (True, False): 2,
        }
        return tabela_severidade[(critico, moderado)]

    def analisar_distribuicao(
        self,
        base_referencia: np.ndarray,
        base_producao: np.ndarray,
    ) -> ResultadoDrift:
        psi = self.calcular_psi(base_referencia, base_producao)
        ks_stat, ks_pval = self.calcular_ks(base_referencia, base_producao)
        wasserstein = self.calcular_wasserstein(base_referencia, base_producao)
        status = self.classificar_severidade(psi, ks_pval)

        return ResultadoDrift(
            psi=psi,
            ks_estatistica=ks_stat,
            ks_pvalor=ks_pval,
            wasserstein_distancia=wasserstein,
            status_severidade=status,
        )
