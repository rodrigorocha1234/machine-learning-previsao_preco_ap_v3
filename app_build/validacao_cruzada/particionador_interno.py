from typing import Final

from sklearn.model_selection import KFold


class ParticionadorInterno:
    def __init__(
        self, splits: int = 5, embaralhar: bool = True, semente: int = 42
    ) -> None:
        self._splits: Final[int] = splits
        self._embaralhar: Final[bool] = embaralhar
        self._semente: Final[int] = semente

    def obter_validador(self) -> KFold:
        return KFold(
            n_splits=self._splits,
            shuffle=self._embaralhar,
            random_state=self._semente,
        )
