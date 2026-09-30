from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class ResultadoShapiro:
    nome_modelo: str
    estatistica: float
    p_valor: float
    eh_normal: bool


@dataclass(frozen=True)
class ResultadoFriedman:
    estatistica: float
    p_valor: float
    eh_significativo: bool
    ranks_medios: dict[str, float]


@dataclass(frozen=True)
class ComparacaoParNemenyi:
    modelo_a: str
    modelo_b: str
    diferenca_ranks: float
    diferenca_critica: float
    eh_significativo: bool


@dataclass(frozen=True)
class ResultadoNemenyi:
    diferenca_critica: float
    comparacoes: tuple[ComparacaoParNemenyi, ...]


@runtime_checkable
class ContratoEstatistica(Protocol):
    def avaliar_residuos(
        self, nome_modelo: str, residuos: pd.Series
    ) -> ResultadoShapiro: ...

    def executar_friedman(self, matriz_scores: pd.DataFrame) -> ResultadoFriedman: ...

    def executar_nemenyi(
        self, resultado_friedman: ResultadoFriedman, matriz_scores: pd.DataFrame
    ) -> ResultadoNemenyi: ...
