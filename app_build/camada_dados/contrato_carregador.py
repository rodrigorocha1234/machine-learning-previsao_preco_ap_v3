from pathlib import Path
from typing import Final, Protocol, TypeAlias, TypeVar, runtime_checkable

T_co = TypeVar("T_co", covariant=True)
CaminhoEntrada: TypeAlias = Path | str


class CarregamentoDadosErro(Exception):
    def __init__(self, mensagem: str, causa: Exception | None = None) -> None:
        super().__init__(mensagem)
        self.causa: Final[Exception | None] = causa


@runtime_checkable
class ProtocoloCarregador(Protocol[T_co]):
    def carregar(self, caminho_origem: CaminhoEntrada) -> T_co: ...
