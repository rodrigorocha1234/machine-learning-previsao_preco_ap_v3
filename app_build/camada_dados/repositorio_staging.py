from pathlib import Path
from typing import Final

import pandas as pd

from app_build.camada_dados.adaptador_sqlite import AdaptadorSqlite


class StagingRepositorioErro(Exception):
    def __init__(self, mensagem: str, causa: Exception | None = None) -> None:
        super().__init__(mensagem)
        self.causa: Final[Exception | None] = causa


class RepositorioStaging:
    def __init__(self, adaptador: AdaptadorSqlite | None = None) -> None:
        caminho_padrao = Path("storage-data/staging_apartamentos.db")
        self._adaptador: Final[AdaptadorSqlite] = adaptador or AdaptadorSqlite(
            caminho_padrao
        )

    def salvar_snapshot(
        self, dados: pd.DataFrame, nome_tabela: str = "apartamentos_staging"
    ) -> None:
        try:
            with self._adaptador.obter_conexao() as conexao:
                dados.to_sql(nome_tabela, conexao, if_exists="replace", index=False)
        except Exception as erro:
            raise StagingRepositorioErro(
                f"Erro ao salvar staging na tabela {nome_tabela}: {erro}", erro
            ) from erro

    def carregar_snapshot(
        self, nome_tabela: str = "apartamentos_staging"
    ) -> pd.DataFrame:
        try:
            with self._adaptador.obter_conexao() as conexao:
                return pd.read_sql_query(f"SELECT * FROM {nome_tabela}", conexao)
        except Exception as erro:
            raise StagingRepositorioErro(
                f"Erro ao recuperar staging da tabela {nome_tabela}: {erro}", erro
            ) from erro
