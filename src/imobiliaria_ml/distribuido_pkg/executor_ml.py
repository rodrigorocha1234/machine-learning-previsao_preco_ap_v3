"""Contrato genérico para executores de Machine Learning."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import numpy as np
import pandas as pd
from ..modelos_pkg.regressor_protocol import RegressorProtocol

TModelo = TypeVar("TModelo", bound=RegressorProtocol)
TDados = TypeVar("TDados", pd.DataFrame, np.ndarray)
TAlvo = TypeVar("TAlvo", pd.Series, np.ndarray)


class ExecutorML(Generic[TModelo, TDados, TAlvo], ABC):
    """Executor abstrato genérico que preserva tipos de modelo e dados."""

    @abstractmethod
    def ajustar(
        self,
        modelo: TModelo,
        dados: TDados,
        alvo: TAlvo,
    ) -> TModelo:
        """Ajusta o estimador retornando a própria instância tipada."""
        ...

    @abstractmethod
    def prever(
        self,
        modelo: TModelo,
        dados: TDados,
    ) -> np.ndarray:
        """Produz predições a partir dos dados fornecidos."""
        ...
