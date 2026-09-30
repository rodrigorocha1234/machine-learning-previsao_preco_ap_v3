import sqlite3
from pathlib import Path
from typing import Final


class AdaptadorSqliteErro(Exception):
    def __init__(self, mensagem: str, causa: Exception | None = None) -> None:
        super().__init__(mensagem)
        self.causa: Final[Exception | None] = causa


class AdaptadorSqlite:
    def __init__(self, caminho_banco: Path | str) -> None:
        self._caminho: Final[Path] = Path(caminho_banco)

    def obter_conexao(self) -> sqlite3.Connection:
        try:
            self._caminho.parent.mkdir(parents=True, exist_ok=True)
            return sqlite3.connect(str(self._caminho))
        except Exception as erro:
            raise AdaptadorSqliteErro(
                f"Erro ao conectar ao banco SQLite {self._caminho}: {erro}", erro
            ) from erro
