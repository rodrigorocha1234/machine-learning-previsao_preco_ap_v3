from dataclasses import dataclass
from typing import Final

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ParticaoDados:
    desenvolvimento: pd.DataFrame
    holdout: pd.DataFrame


class DivisorEstratificado:
    def __init__(self, proporcao_holdout: float = 0.20, semente: int = 42) -> None:
        self._proporcao: Final[float] = proporcao_holdout
        self._semente: Final[int] = semente

    def dividir(self, dados: pd.DataFrame) -> ParticaoDados:
        base_desenvolvimento, base_holdout = train_test_split(
            dados,
            test_size=self._proporcao,
            random_state=self._semente,
            shuffle=True,
        )
        return ParticaoDados(
            desenvolvimento=base_desenvolvimento.copy().reset_index(drop=True),
            holdout=base_holdout.copy().reset_index(drop=True),
        )
