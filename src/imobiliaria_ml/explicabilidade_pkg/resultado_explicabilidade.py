"""Estrutura do resultado da explicabilidade do modelo."""

from dataclasses import dataclass, field
import pandas as pd


@dataclass(frozen=True)
class ResultadoExplicabilidade:
    """Consolidação dos artefatos de interpretação e relevância de atributos."""

    tipo_modelo: str
    intercepto: float | None = None
    tabela_coeficientes: pd.DataFrame = field(default_factory=pd.DataFrame)
    equacao_texto: str = ""
    interpretacao_texto: str = ""
    tabela_importancias: pd.DataFrame = field(default_factory=pd.DataFrame)
