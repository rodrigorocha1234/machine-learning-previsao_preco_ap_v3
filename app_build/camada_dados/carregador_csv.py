from pathlib import Path
from typing import override

import pandas as pd

from app_build.camada_dados.contrato_carregador import (
    CaminhoEntrada,
    CarregamentoDadosErro,
    ProtocoloCarregador,
)


class CarregadorCsv(ProtocoloCarregador[pd.DataFrame]):
    @override
    def carregar(self, caminho_origem: CaminhoEntrada) -> pd.DataFrame:
        try:
            return pd.read_csv(Path(caminho_origem))
        except Exception as erro:
            raise CarregamentoDadosErro(
                f"Erro ao carregar CSV {caminho_origem}: {erro}", erro
            ) from erro
