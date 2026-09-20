"""Carregador de planilhas Excel."""

from pathlib import Path
import pandas as pd
from .carregador_base import CarregadorBase


class CarregadorExcel(CarregadorBase):
    """Estratégia de carregamento para planilhas Excel."""

    def carregar_dados(self, origem: str | Path) -> pd.DataFrame:
        """Carrega uma planilha Excel para DataFrame.

        Parameters
        ----------
        origem : str | Path
            Caminho do arquivo Excel (.xlsx ou .xls).

        Returns
        -------
        pd.DataFrame
            DataFrame carregado.
        """
        caminho = Path(origem)
        if not caminho.exists():
            raise FileNotFoundError(f"Arquivo Excel não encontrado: {caminho}")
        return pd.read_excel(caminho)
