"""Implementação local do ExecutorML baseada em scikit-learn."""

from typing import TypeVar
import numpy as np
import pandas as pd
from ..modelos_pkg.regressor_protocol import RegressorProtocol
from .executor_ml import ExecutorML

TModelo = TypeVar("TModelo", bound=RegressorProtocol)
TDados = TypeVar("TDados", pd.DataFrame, np.ndarray)
TAlvo = TypeVar("TAlvo", pd.Series, np.ndarray)


class ExecutorSklearn(ExecutorML[TModelo, TDados, TAlvo]):
    """Executor que executa fit e predict localmente no scikit-learn."""

    def ajustar(
        self,
        modelo: TModelo,
        dados: TDados,
        alvo: TAlvo,
    ) -> TModelo:
        """Ajusta o estimador localmente preservando seu tipo."""
        modelo.fit(dados, alvo)
        return modelo

    def prever(
        self,
        modelo: TModelo,
        dados: TDados,
    ) -> np.ndarray:
        """Executa a predição local."""
        return np.asarray(modelo.predict(dados), dtype=float)
