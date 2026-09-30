from typing import Protocol, runtime_checkable

import pandas as pd


class ViolacaoIsolamentoDadosErro(Exception):
    def __init__(self, mensagem: str) -> None:
        super().__init__(mensagem)


@runtime_checkable
class ContratoIsolador(Protocol):
    def bloquear(self, dados_holdout: pd.DataFrame) -> None: ...

    def liberar_holdout(self, chave_autorizacao: str) -> pd.DataFrame: ...
