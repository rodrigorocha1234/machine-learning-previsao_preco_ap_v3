from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import pandas as pd


class StatusAmostral(Enum):
    SUFICIENTE = "SUFICIENTE"
    AMOSTRA_INSUFICIENTE = "AMOSTRA_INSUFICIENTE"
    NAO_DISPONIVEL = "NAO_DISPONIVEL"


@dataclass(frozen=True)
class EstatisticasNivel:
    nivel: str
    nome_chave: str
    media_valor: float
    mediana_valor: float
    media_valor_m2: float
    mediana_valor_m2: float
    desvio_valor_m2: float
    total_amostras: int
    status_amostral: StatusAmostral


@dataclass(frozen=True)
class ResultadoImobiliario:
    valor_previsto: float
    valor_m2_previsto: float
    estatisticas_global: EstatisticasNivel
    estatisticas_zona: EstatisticasNivel
    estatisticas_bairro: EstatisticasNivel
    indice_imovel_global: float
    indice_imovel_zona: float
    indice_imovel_bairro: float
    indice_zona_global: float
    indice_bairro_zona: float
    indice_bairro_global: float
    diferenca_percentual_global: float
    diferenca_percentual_zona: float
    diferenca_percentual_bairro: float
    nivel_referencia_utilizado: str
    status_final_amostra: StatusAmostral
    desconto_moderado_5: float
    desconto_agressivo_10: float
    desconto_queima_15: float
    faixa_segura_piso: float
    faixa_segura_teto: float


@runtime_checkable
class ContratoNegocio(Protocol):
    def enriquecer_previsao(
        self,
        valor_previsto: float,
        metragem: float,
        zona: str,
        bairro: str,
    ) -> ResultadoImobiliario: ...

    def enriquecer_dataframe(
        self,
        dados_x: pd.DataFrame,
        vetor_previsoes: pd.Series,
    ) -> pd.DataFrame: ...
