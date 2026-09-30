from typing import Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class ContratoEstimador(Protocol):
    def fit(
        self, x_dados: pd.DataFrame | np.ndarray, y_vetor: pd.Series | np.ndarray
    ) -> Self: ...

    def predict(self, x_dados: pd.DataFrame | np.ndarray) -> np.ndarray: ...
