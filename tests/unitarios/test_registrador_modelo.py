"""Testes unitários para o RegistradorModelo."""

from unittest.mock import MagicMock
import mlflow
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from imobiliaria_ml.mlflow_pkg.registrador_modelo import RegistradorModelo


def test_registrador_modelo_nao_grava_code_paths(monkeypatch) -> None:
    """Garante que código python não é enviado via code_paths ao registrar modelo."""
    chamadas_log_model: list[dict[str, object]] = []

    class MockInfoModelo:
        registered_model_version = "1"
        model_uri = "models:/preco-imoveis-teste/1"

    def mock_log_model(**kwargs: object) -> MockInfoModelo:
        chamadas_log_model.append(kwargs)
        return MockInfoModelo()

    class MockClient:
        def __init__(self, tracking_uri=None) -> None:
            pass

        def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
            pass

        def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
            pass

    monkeypatch.setattr(mlflow.pyfunc, "log_model", mock_log_model)
    monkeypatch.setattr(mlflow, "active_run", lambda: None)
    monkeypatch.setattr("imobiliaria_ml.mlflow_pkg.registrador_modelo.MlflowClient", MockClient)

    registrador = RegistradorModelo(nome_modelo="preco-imoveis-teste", tracking_uri="http://mock:5000")
    pipeline = Pipeline([("modelo", Ridge())])

    uri = registrador.registrar_modelo_campeao(
        pipeline_campea=pipeline,
        metricas={"rmse": 10000.0},
        desconto_maximo=10.0,
        nome_modelo_concreto="ridge",
    )

    assert len(chamadas_log_model) == 1
    args_passados = chamadas_log_model[0]

    # Verifica que code_paths NÃO foi passado ou é None
    assert "code_paths" not in args_passados or args_passados.get("code_paths") is None
    assert args_passados.get("artifact_path") == "modelo_imobiliario"
    assert args_passados.get("registered_model_name") == "preco-imoveis-teste"


def test_registrador_modelo_com_usar_votacao_adiciona_sufixo(monkeypatch) -> None:
    """Verifica que com usar_votacao=True o sufixo _votacao é adicionado ao runName e às tags."""
    tags_capturadas: dict[str, str] = {}
    tags_versao_capturadas: dict[str, str] = {}

    class MockRunInfo:
        run_id = "run_campeao_123"

    class MockRun:
        info = MockRunInfo()

    class MockInfoModelo:
        registered_model_version = "2"
        model_uri = "models:/preco-imoveis-teste/2"

    class MockClient:
        def __init__(self, tracking_uri=None) -> None:
            pass

        def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
            pass

        def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
            tags_versao_capturadas[key] = value

    monkeypatch.setattr(mlflow.pyfunc, "log_model", lambda **kwargs: MockInfoModelo())
    monkeypatch.setattr(mlflow, "active_run", lambda: MockRun())
    monkeypatch.setattr(mlflow, "set_tags", lambda t: tags_capturadas.update(t))
    monkeypatch.setattr(mlflow, "end_run", lambda: None)
    monkeypatch.setattr("imobiliaria_ml.mlflow_pkg.registrador_modelo.MlflowClient", MockClient)

    registrador = RegistradorModelo(
        nome_modelo="preco-imoveis-teste",
        tracking_uri="http://mock:5000",
        usar_votacao=True,
    )
    pipeline = Pipeline([("modelo", Ridge())])

    registrador.registrar_modelo_campeao(
        pipeline_campea=pipeline,
        metricas={"rmse": 10000.0},
        desconto_maximo=10.0,
        nome_modelo_concreto="ridge",
    )

    assert tags_capturadas.get("mlflow.runName") == "campeao_ridge_votacao"
    assert tags_capturadas.get("usar_votacao") == "true"
    assert tags_versao_capturadas.get("usar_votacao") == "true"


def test_registrador_modelo_grava_tags_de_negocio(monkeypatch) -> None:
    """Valida que métricas de negócio são gravadas como tags no Model Registry."""
    from imobiliaria_ml.negocio_pkg.metricas_negocio_resultado import MetricasNegocioResultado

    tags_capturadas: dict[str, str] = {}
    tags_versao_capturadas: dict[str, str] = {}

    class MockRunInfo:
        run_id = "run_campeao_negocio"

    class MockRun:
        info = MockRunInfo()

    class MockInfoModelo:
        registered_model_version = "3"
        model_uri = "models:/preco-imoveis-teste/3"

    class MockClient:
        def __init__(self, tracking_uri=None) -> None:
            pass

        def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
            pass

        def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
            tags_versao_capturadas[key] = value

    monkeypatch.setattr(mlflow.pyfunc, "log_model", lambda **kwargs: MockInfoModelo())
    monkeypatch.setattr(mlflow, "active_run", lambda: MockRun())
    monkeypatch.setattr(mlflow, "set_tags", lambda t: tags_capturadas.update(t))
    monkeypatch.setattr(mlflow, "end_run", lambda: None)
    monkeypatch.setattr("imobiliaria_ml.mlflow_pkg.registrador_modelo.MlflowClient", MockClient)

    registrador = RegistradorModelo(
        nome_modelo="preco-imoveis-teste",
        tracking_uri="http://mock:5000",
    )
    pipeline = Pipeline([("modelo", Ridge())])

    metricas_neg = MetricasNegocioResultado(
        cobertura_5=0.85,
        cobertura_10=0.96,
        cobertura_15=0.99,
        mae_reais=11500.0,
        vies_medio=250.0,
        desconto_seguro_recomendado=6.0,
        risco_subprecificacao=0.01,
        risco_superprecificacao=0.01,
        margem_negociacao_estimada=30000.0,
        receita_potencial_perdida=15000.0,
    )

    registrador.registrar_modelo_campeao(
        pipeline_campea=pipeline,
        metricas={"rmse": 9500.0},
        desconto_maximo=10.0,
        nome_modelo_concreto="ridge",
        metricas_negocio=metricas_neg,
    )

    assert tags_versao_capturadas.get("negocio_desconto_seguro") == "6.00%"
    assert tags_versao_capturadas.get("negocio_cobertura_10") == "96.00%"
    assert tags_versao_capturadas.get("negocio_cobertura_5") == "85.00%"
    assert tags_versao_capturadas.get("negocio_mae_reais") == "R$ 11,500.00"
    assert tags_versao_capturadas.get("negocio_margem_negociacao") == "R$ 30,000.00"
    assert tags_capturadas.get("desconto_seguro") == "6.00%"
    assert tags_capturadas.get("cobertura_10") == "96.00%"


