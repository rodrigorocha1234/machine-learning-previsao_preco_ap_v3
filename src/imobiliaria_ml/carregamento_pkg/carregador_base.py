"""Interface base para estratégias de carregamento de dados."""

from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd


class CarregadorBase(ABC):
    """Classe base abstrata para estratégias de carregamento."""

    @abstractmethod
    def carregar_dados(self, origem: str | Path) -> pd.DataFrame:
        """Carrega dados da fonte especificada para um DataFrame.

        Parameters
        ----------
        origem : str | Path
            Caminho do arquivo ou string de conexão do banco de dados.

        Returns
        -------
        pd.DataFrame
            DataFrame carregado com os dados.
        """
        ...
