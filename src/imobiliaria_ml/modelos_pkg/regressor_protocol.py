"""Protocol comportamental para regressores da solução."""

from typing import Protocol
import numpy as np
import pandas as pd


class RegressorProtocol(Protocol):
    """Protocolo comportamental comum a todos os regressores."""

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "RegressorProtocol":
        """Ajusta o estimador com as características e o alvo."""
        ...

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Produz estimativas numéricas a partir das características."""
        ...
