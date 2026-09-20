"""Carregador de arquivos CSV."""

from pathlib import Path
import pandas as pd
from .carregador_base import CarregadorBase


class CarregadorCsv(CarregadorBase):
    """Estratégia de carregamento para arquivos CSV."""

    def carregar_dados(self, origem: str | Path) -> pd.DataFrame:
        """Carrega um arquivo CSV para DataFrame.

        Parameters
        ----------
        origem : str | Path
            Caminho do arquivo CSV.

        Returns
        -------
        pd.DataFrame
            DataFrame carregado.
        """
        caminho = Path(origem)
        if not caminho.exists():
            raise FileNotFoundError(f"Arquivo CSV não encontrado: {caminho}")
        return pd.read_csv(caminho)
