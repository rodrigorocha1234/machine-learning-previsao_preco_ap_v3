from typing import Final

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold


class ParticionadorExterno:
    def __init__(self, splits: int = 5, repeticoes: int = 3, semente: int = 42) -> None:
        self._splits: Final[int] = splits
        self._repeticoes: Final[int] = repeticoes
        self._semente: Final[int] = semente

    def gerar_divisoes(
        self, dados_x: pd.DataFrame
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        rkf = RepeatedKFold(
            n_splits=self._splits,
            n_repeats=self._repeticoes,
            random_state=self._semente,
        )
        return tuple(
            (idx_treino, idx_teste) for idx_treino, idx_teste in rkf.split(dados_x)
        )
