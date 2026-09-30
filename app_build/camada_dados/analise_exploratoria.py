from dataclasses import dataclass
from typing import Final

import pandas as pd


@dataclass(frozen=True)
class RelatorioExploratorio:
    quantidade_registros: int
    estatisticas_gerais: pd.DataFrame
    estatisticas_zona: pd.DataFrame
    estatisticas_bairro: pd.DataFrame
    contagem_zona: pd.Series
    contagem_bairro: pd.Series


class AnaliseExploratoria:
    def __init__(self, target: str = "Valor_da_Venda") -> None:
        self._target: Final[str] = target
        self._colunas_numericas: Final[tuple[str, ...]] = (
            "Metragem",
            "Quartos",
            "Banheiros",
            "Vagas_Garagem",
            target,
        )

    def executar(self, dados: pd.DataFrame) -> RelatorioExploratorio:
        cols_presentes = list(
            filter(lambda col: col in dados.columns, self._colunas_numericas)
        )
        estatisticas_gerais = dados[cols_presentes].describe().T

        estatisticas_zona = dados.groupby("Zona")[cols_presentes].agg(
            ["mean", "median", "std"]
        )

        estatisticas_bairro = dados.groupby("Bairro")[cols_presentes].agg(
            ["mean", "median", "count"]
        )

        contagem_zona = dados["Zona"].value_counts()
        contagem_bairro = dados["Bairro"].value_counts()

        return RelatorioExploratorio(
            quantidade_registros=len(dados),
            estatisticas_gerais=estatisticas_gerais,
            estatisticas_zona=estatisticas_zona,
            estatisticas_bairro=estatisticas_bairro,
            contagem_zona=contagem_zona,
            contagem_bairro=contagem_bairro,
        )
