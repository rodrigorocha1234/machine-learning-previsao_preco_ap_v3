from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd
from sklearn.base import BaseEstimator

from app_build.ajuste_modelos.explicador_parametros import ExplicacaoParametro
from app_build.estatistica_modelos.contrato_estatistica import (
    ResultadoFriedman,
    ResultadoNemenyi,
    ResultadoShapiro,
)
from app_build.regras_negocio.agregador_hierarquico import EstatisticasHierarquicas
from app_build.regras_negocio.contrato_negocio import ResultadoImobiliario
from app_build.regras_negocio.motor_imobiliario import MotorImobiliario
from app_build.selecao_modelos.contrato_seletor import DecisaoSelecao
from app_build.validacao_cruzada.contrato_validador import (
    MetricasRegressao,
    ResultadoNestedCv,
)


@dataclass(frozen=True)
class EventoNestedCvConcluido:
    nome_modelo: str
    resultado_cv: ResultadoNestedCv
    explicacoes_parametros: tuple[ExplicacaoParametro, ...]


@dataclass(frozen=True)
class EventoEstatisticaConcluida:
    resultado_friedman: ResultadoFriedman
    resultado_nemenyi: ResultadoNemenyi
    resultados_shapiro: tuple[ResultadoShapiro, ...]


@dataclass(frozen=True)
class EventoModeloSelecionado:
    decisao: DecisaoSelecao


@dataclass(frozen=True)
class EventoTreinoFinalConcluido:
    nome_modelo: str
    estimador: BaseEstimator
    explicacoes_parametros: tuple[ExplicacaoParametro, ...]
    dados_exemplo: pd.DataFrame
    motor_imobiliario: MotorImobiliario | None = None
    parametros_componentes: dict[str, dict[str, object]] = field(default_factory=dict)



@dataclass(frozen=True)
class EventoHoldoutAvaliado:
    metricas_global: MetricasRegressao
    metricas_zona: dict[str, MetricasRegressao]
    metricas_bairro: dict[str, MetricasRegressao]


@dataclass(frozen=True)
class EventoRegrasNegocioConcluido:
    estatisticas_hierarquicas: EstatisticasHierarquicas
    dados_enriquecidos: pd.DataFrame
    exemplos_simulacao: tuple[ResultadoImobiliario, ...]
    metricas_negocio: dict[str, float]


@runtime_checkable
class ContratoObservador(Protocol):
    def ao_concluir_nested_cv(self, evento: EventoNestedCvConcluido) -> None: ...

    def ao_concluir_estatistica(self, evento: EventoEstatisticaConcluida) -> None: ...

    def ao_selecionar_modelo(self, evento: EventoModeloSelecionado) -> None: ...

    def ao_concluir_treino_final(self, evento: EventoTreinoFinalConcluido) -> None: ...

    def ao_avaliar_holdout(self, evento: EventoHoldoutAvaliado) -> None: ...

    def ao_concluir_regras_negocio(
        self, evento: EventoRegrasNegocioConcluido
    ) -> None: ...
