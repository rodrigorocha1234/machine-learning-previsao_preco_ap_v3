"""Resultado final do treinamento e avaliação preservando o tipo do modelo."""

from dataclasses import dataclass
from typing import Generic, TypeVar
from ..modelos_pkg.regressor_protocol import RegressorProtocol
from ..validacao_pkg.metricas_regressao import MetricasRegressao
from ..negocio_pkg.metricas_negocio_resultado import MetricasNegocioResultado
from ..explicabilidade_pkg.resultado_explicabilidade import ResultadoExplicabilidade

TModelo = TypeVar("TModelo", bound=RegressorProtocol)


@dataclass(frozen=True)
class ResultadoTreinamento(Generic[TModelo]):
    """Estrutura genérica contendo o modelo campeão treinado e suas avaliações."""

    modelo: TModelo
    metricas: MetricasRegressao
    nome_campeao: str = ""
    metricas_negocio: MetricasNegocioResultado | None = None
    explicabilidade: ResultadoExplicabilidade | None = None
    uri_registro: str = ""

