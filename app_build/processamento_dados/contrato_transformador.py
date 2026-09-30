from typing import Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class ContratoTransformador(Protocol):
    def ajustar(
        self, dados_x: pd.DataFrame, vetor_y: pd.Series | None = None
    ) -> Self: ...

    def transformar(
        self, dados_x: pd.DataFrame
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]: ...

    def ajustar_transformar(
        self, dados_x: pd.DataFrame, vetor_y: pd.Series | None = None
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]: ...
