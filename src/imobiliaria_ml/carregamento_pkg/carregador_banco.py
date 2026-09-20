"""Carregador de tabelas de bancos relacionais."""

from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine
from .carregador_base import CarregadorBase


class CarregadorBanco(CarregadorBase):
    """Estratégia de carregamento para bancos relacionais."""

    def __init__(self, consulta_sql: str = "SELECT * FROM imoveis") -> None:
        """Inicializa com a consulta SQL desejada.

        Parameters
        ----------
        consulta_sql : str
            Query SQL para selecionar os registros.
        """
        self._consulta_sql = consulta_sql

    def carregar_dados(self, origem: str | Path) -> pd.DataFrame:
        """Carrega dados executando a consulta no banco de dados.

        Parameters
        ----------
        origem : str | Path
            String de conexão SQLAlchemy (ex: 'sqlite:///dados.db' ou 'postgresql://...').

        Returns
        -------
        pd.DataFrame
            DataFrame carregado.
        """
        engine = create_engine(str(origem))
        with engine.connect() as conexao:
            return pd.read_sql_query(self._consulta_sql, con=conexao)
