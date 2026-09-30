from pathlib import Path
from typing import Final

from app_build.configuracao_sistema.contrato_configuracao import (
    CaminhoArquivo,
    ConfiguracaoGeral,
    ContratoConfiguracao,
)
from app_build.configuracao_sistema.leitor_yaml import LeitorYaml
from app_build.configuracao_sistema.validador_esquema import ValidadorEsquema


class ArmazemConfiguracao:
    def __init__(
        self,
        leitor: ContratoConfiguracao | None = None,
        validador: ValidadorEsquema | None = None,
    ) -> None:
        self._leitor: Final[ContratoConfiguracao] = leitor or LeitorYaml()
        self._validador: Final[ValidadorEsquema] = validador or ValidadorEsquema()
        self._configuracao_geral: ConfiguracaoGeral | None = None
        self._modelos: dict[str, dict[str, object]] = {}

    def inicializar(
        self,
        caminho_pipeline: CaminhoArquivo = Path("configs/pipeline.yaml"),
        caminho_modelos: CaminhoArquivo = Path("configs/modelos.yaml"),
    ) -> None:
        geral = self._leitor.carregar_geral(caminho_pipeline)
        self._validador.validar(geral)
        self._configuracao_geral = geral
        self._modelos = self._leitor.carregar_modelos(caminho_modelos)

    @property
    def geral(self) -> ConfiguracaoGeral:
        assert self._configuracao_geral is not None, (
            "ArmazemConfiguracao nao foi inicializado."
        )
        return self._configuracao_geral

    @property
    def modelos(self) -> dict[str, dict[str, object]]:
        return self._modelos
