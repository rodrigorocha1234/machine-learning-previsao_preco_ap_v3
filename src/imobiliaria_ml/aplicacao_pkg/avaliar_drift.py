"""Ponto de entrada para avaliação e monitoramento de Data Drift."""

from pathlib import Path
import pandas as pd
from ..configuracao_pkg.leitor_configuracao import LeitorConfiguracao
from ..drift_pkg.monitor_drift import MonitorDrift
from ..drift_pkg.resultado_drift import ResultadoDrift


class AvaliarDrift:
    """Executável para auditoria de desvio de dados entre produção e baseline."""

    def __init__(self, caminho_configuracao: str | Path = "configuracao.yaml") -> None:
        self._leitor = LeitorConfiguracao()
        config = self._leitor.carregar_do_arquivo(caminho_configuracao)
        self._monitor = MonitorDrift(
            psi_limite_atencao=config.drift_psi_atencao,
            psi_limite_forte=config.drift_psi_forte,
        )

    def avaliar_arquivos(
        self,
        caminho_base: str | Path,
        caminho_novo: str | Path,
        colunas_numericas: list[str] | None = None,
        colunas_categoricas: list[str] | None = None,
    ) -> ResultadoDrift:
        """Executa a checagem de drift lendo dois arquivos CSV.

        Parameters
        ----------
        caminho_base : str | Path
            Base de referência histórica.
        caminho_novo : str | Path
            Novos dados recebidos.
        colunas_numericas : list[str] | None
            Lista de features numéricas.
        colunas_categoricas : list[str] | None
            Lista de features categóricas.

        Returns
        -------
        ResultadoDrift
            Diagnóstico consolidado de drift.
        """
        df_base = pd.read_csv(caminho_base)
        df_novo = pd.read_csv(caminho_novo)

        cols_num = colunas_numericas or ["Quartos", "Banheiros", "Vagas", "Metragem"]
        cols_cat = colunas_categoricas or ["Zona"]

        return self._monitor.avaliar(
            df_base=df_base,
            df_atual=df_novo,
            colunas_numericas=cols_num,
            colunas_categoricas=cols_cat,
        )


if __name__ == "__main__":
    app = AvaliarDrift()
