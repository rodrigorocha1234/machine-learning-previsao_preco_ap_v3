from typing import Self, override

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ExtratorAtributos(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass

    @override
    def fit(self, x_dados: pd.DataFrame, y_vetor: pd.Series | None = None) -> Self:
        return self

    @override
    def transform(self, x_dados: pd.DataFrame) -> pd.DataFrame:
        quartos_seguros = x_dados["Quartos"].clip(lower=1)
        transformado = x_dados.assign(
            razao_banheiros_quartos=x_dados["Banheiros"] / quartos_seguros,
            metragem_por_quarto=x_dados["Metragem"] / quartos_seguros,
            total_comodos=x_dados["Quartos"]
            + x_dados["Banheiros"]
            + x_dados["Vagas_Garagem"],
        )
        return transformado
