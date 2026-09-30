from pathlib import Path
from typing import override

import pandas as pd

from app_build.camada_dados.contrato_carregador import (
    CaminhoEntrada,
    CarregamentoDadosErro,
    ProtocoloCarregador,
)


class CarregadorExcel(ProtocoloCarregador[pd.DataFrame]):
    @override
    def carregar(self, caminho_origem: CaminhoEntrada) -> pd.DataFrame:
        try:
            caminho = Path(caminho_origem)
            assert caminho.exists(), f"Arquivo nao encontrado: {caminho}"
            dados = pd.read_excel(caminho)
            return dados
        except Exception as erro:
            raise CarregamentoDadosErro(
                f"Falha ao carregar dados do Excel em {caminho_origem}: {erro}", erro
            ) from erro
