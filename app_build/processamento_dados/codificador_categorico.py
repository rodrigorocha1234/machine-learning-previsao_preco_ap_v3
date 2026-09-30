from typing import Final

from sklearn.preprocessing import OneHotEncoder


class CodificadorCategorico:
    def __init__(self) -> None:
        self._encoder: Final[OneHotEncoder] = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

    def obter_encoder(self) -> OneHotEncoder:
        return self._encoder
