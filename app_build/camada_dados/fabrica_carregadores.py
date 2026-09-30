from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

import pandas as pd

from app_build.camada_dados.carregador_csv import CarregadorCsv
from app_build.camada_dados.carregador_excel import CarregadorExcel
from app_build.camada_dados.carregador_parquet import CarregadorParquet
from app_build.camada_dados.contrato_carregador import (
    CaminhoEntrada,
    CarregamentoDadosErro,
    ProtocoloCarregador,
)

CriadorCarregador = Callable[[], ProtocoloCarregador[pd.DataFrame]]

TABELA_CARREGADORES: Final[Mapping[str, CriadorCarregador]] = {
    ".xlsx": CarregadorExcel,
    ".xls": CarregadorExcel,
    ".csv": CarregadorCsv,
    ".parquet": CarregadorParquet,
}


class FabricaCarregadores:
    @staticmethod
    def obter_carregador(
        caminho_arquivo: CaminhoEntrada,
    ) -> ProtocoloCarregador[pd.DataFrame]:
        extensao = Path(caminho_arquivo).suffix.lower()
        try:
            criador = TABELA_CARREGADORES[extensao]
            return criador()
        except KeyError as erro:
            raise CarregamentoDadosErro(
                f"Extensao de arquivo nao suportada: {extensao}", erro
            ) from erro
