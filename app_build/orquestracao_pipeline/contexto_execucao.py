from typing import Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from app_build.configuracao_sistema.contrato_configuracao import ConfiguracaoGeral
from app_build.estatistica_modelos.contrato_estatistica import (
    ResultadoFriedman,
    ResultadoNemenyi,
    ResultadoShapiro,
)
from app_build.isolamento_dados.cofre_holdout import CofreHoldout
from app_build.observabilidade_metricas.coletor_prometheus import ColetorPrometheus
from app_build.rastreamento_mlflow.despachante_eventos import DespachanteEventos
from app_build.regras_negocio.motor_imobiliario import MotorImobiliario
from app_build.selecao_modelos.contrato_seletor import DecisaoSelecao
from app_build.validacao_cruzada.contrato_validador import (
    MetricasRegressao,
    ResultadoNestedCv,
)


class ContextoExecucao:
    def __init__(self, coletor: ColetorPrometheus | None = None) -> None:
        self.configuracao_geral: ConfiguracaoGeral | None = None
        self.configuracao_modelos: dict[str, dict[str, object]] = {}
        self.dados_brutos: pd.DataFrame | None = None
        self.dados_desenvolvimento: pd.DataFrame | None = None
        self.cofre_holdout: CofreHoldout = CofreHoldout()
        self.resultados_nested_cv: dict[str, ResultadoNestedCv] = {}
        self.resultado_friedman: ResultadoFriedman | None = None
        self.resultado_nemenyi: ResultadoNemenyi | None = None
        self.resultados_shapiro: list[ResultadoShapiro] = []
        self.decisao_selecao: DecisaoSelecao | None = None
        self.modelo_campeao_final: BaseEstimator | None = None
        self.parametros_componentes: dict[str, dict[str, object]] = {}
        self.motor_imobiliario: MotorImobiliario | None = None
        self.metricas_holdout_global: MetricasRegressao | None = None
        self.metricas_holdout_zona: dict[str, MetricasRegressao] = {}
        self.metricas_holdout_bairro: dict[str, MetricasRegressao] = {}
        self.estado_congelado: bool = False
        self.despachante: Final[DespachanteEventos] = DespachanteEventos()
        self.dados_holdout_liberados: pd.DataFrame | None = None
        self.divisoes_externas: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None
        self.coletor: ColetorPrometheus | None = coletor
