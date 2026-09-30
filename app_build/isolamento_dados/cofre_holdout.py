from typing import Final, override

import pandas as pd

from app_build.isolamento_dados.contrato_isolador import (
    ContratoIsolador,
    ViolacaoIsolamentoDadosErro,
)


class CofreHoldout(ContratoIsolador):
    CHAVE_MESTRA: Final[str] = "ESTADO_CONGELADO_CHAVE_LIBERADA"

    def __init__(self) -> None:
        self._dados: pd.DataFrame | None = None
        self._acessado: bool = False

    @override
    def bloquear(self, dados_holdout: pd.DataFrame) -> None:
        assert self._dados is None, "Holdout ja foi bloqueado anteriormente."
        self._dados = dados_holdout.copy()

    @override
    def liberar_holdout(self, chave_autorizacao: str) -> pd.DataFrame:
        try:
            assert self._dados is not None, "Cofre vazio: nenhum holdout foi bloqueado."
            assert not self._acessado, (
                "Violacao: Holdout so pode ser acessado uma unica vez na avaliacao final."
            )
            assert chave_autorizacao == self.CHAVE_MESTRA, (
                "Chave de autorizacao invalida para liberacao do Holdout."
            )
        except AssertionError as falha:
            raise ViolacaoIsolamentoDadosErro(
                f"Tentativa nao autorizada de abertura do Holdout: {falha}"
            ) from falha

        self._acessado = True
        return self._dados.copy()
