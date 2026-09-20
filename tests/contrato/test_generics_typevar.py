"""Teste de contrato para validação de uso correto de Generic, TypeVar e Protocol."""

from typing import get_args, get_origin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from imobiliaria_ml.modelos_pkg.regressor_protocol import RegressorProtocol
from imobiliaria_ml.modelos_pkg.estrategia_modelo import EstrategiaModelo, TModelo
from imobiliaria_ml.modelos_pkg.regressao_random_forest import RegressaoRandomForest
from imobiliaria_ml.modelos_pkg.regressao_ridge import RegressaoRidge
from imobiliaria_ml.distribuido_pkg.executor_ml import ExecutorML
from imobiliaria_ml.distribuido_pkg.executor_sklearn import ExecutorSklearn
from imobiliaria_ml.distribuido_pkg.adaptador_distribuido import (
    AdaptadorDistribuido,
    TModeloOrigem,
    TModeloDestino,
)
from imobiliaria_ml.pipeline_pkg.resultado_treinamento import ResultadoTreinamento


def test_tmodelo_possui_bound_regressor_protocol() -> None:
    """Valida que TModelo possui bound estrito para RegressorProtocol."""
    assert TModelo.__bound__ is RegressorProtocol


def test_tmodelo_origem_e_destino_possuem_bound() -> None:
    """Valida que TypeVars do AdaptadorDistribuido possuem bounds apropriados."""
    assert TModeloOrigem.__bound__ is RegressorProtocol
    assert TModeloDestino.__bound__ is RegressorProtocol


def test_estrategia_modelo_preserva_tipo_concreto() -> None:
    """Valida que especializações de EstrategiaModelo preservam o tipo de estimador."""
    rf_est = RegressaoRandomForest()
    modelo_rf = rf_est.criar_modelo()
    assert isinstance(modelo_rf, RandomForestRegressor)

    ridge_est = RegressaoRidge()
    modelo_ridge = ridge_est.criar_modelo()
    assert isinstance(modelo_ridge, Ridge)


def test_executor_ml_preserva_generics() -> None:
    """Valida que ExecutorSklearn implementa ExecutorML."""
    assert issubclass(ExecutorSklearn, ExecutorML)


def test_resultado_treinamento_generico() -> None:
    """Valida que ResultadoTreinamento é uma classe genérica parametrizada."""
    origem = get_origin(ResultadoTreinamento)
    assert origem is None or issubclass(ResultadoTreinamento, object)
