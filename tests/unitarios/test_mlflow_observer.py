"""Testes unitários para Observer e SujeitoObservavel do MLflow."""

from unittest.mock import MagicMock
from imobiliaria_ml.enums_pkg.tipo_evento import TipoEvento
from imobiliaria_ml.mlflow_pkg.observador import Observador
from imobiliaria_ml.mlflow_pkg.sujeito_observavel import SujeitoObservavel
from imobiliaria_ml.mlflow_pkg.carga_evento import CargaEvento


class ObservadorMock(Observador):
    def __init__(self) -> None:
        self.chamadas: list[TipoEvento] = []

    def atualizar(self, evento: TipoEvento, dados: CargaEvento) -> None:
        self.chamadas.append(evento)


def test_sujeito_notifica_observadores() -> None:
    sujeito = SujeitoObservavel()
    obs = ObservadorMock()
    sujeito.adicionar_observador(obs)

    sujeito.notificar(TipoEvento.EDA_FINALIZADA, CargaEvento(valores={"teste": 1}))
    assert len(obs.chamadas) == 1
    assert obs.chamadas[0] == TipoEvento.EDA_FINALIZADA


def test_observador_mlflow_registra_metricas_na_run_do_modelo(monkeypatch) -> None:
    import mlflow
    from imobiliaria_ml.mlflow_pkg.observador_mlflow import ObservadorMlflow
    import pandas as pd

    metricas_logadas: list[tuple[str, float]] = []
    runs_iniciadas: list[str] = []
    tags_logadas: list[dict[str, str]] = []

    class MockRunInfo:
        def __init__(self, run_id: str) -> None:
            self.run_id = run_id

    class MockRun:
        def __init__(self, run_id: str) -> None:
            self.info = MockRunInfo(run_id)

    run_counter = [0]

    def mock_start_run(run_name=None, nested=False):
        run_counter[0] += 1
        rid = f"run_{run_counter[0]}"
        runs_iniciadas.append(str(run_name))
        return MockRun(rid)

    monkeypatch.setattr(ObservadorMlflow, "_configurar_mlflow", lambda self: None)
    monkeypatch.setattr(mlflow, "start_run", mock_start_run)
    monkeypatch.setattr(mlflow, "end_run", lambda: None)
    monkeypatch.setattr(mlflow, "active_run", lambda: MockRun("pai"))
    monkeypatch.setattr(mlflow, "log_metric", lambda k, v: metricas_logadas.append((k, v)))
    monkeypatch.setattr(mlflow, "set_tags", lambda t: tags_logadas.append(t))
    monkeypatch.setattr(mlflow, "log_param", lambda k, v: None)
    monkeypatch.setattr(mlflow, "log_dict", lambda **kwargs: None)
    monkeypatch.setattr(mlflow, "log_table", lambda **kwargs: None)

    obs = ObservadorMlflow(tracking_uri="http://mock-uri:5000")

    # Inicia GridSearch para o modelo lasso
    obs.atualizar(
        TipoEvento.GRIDSEARCH_INICIADO,
        CargaEvento(valores={"nome_modelo": "lasso"}),
    )
    assert "lasso" in runs_iniciadas
    assert obs._runs_modelos.get("lasso") is not None

    # Finaliza GridSearch
    obs.atualizar(
        TipoEvento.GRIDSEARCH_FINALIZADO,
        CargaEvento(
            valores={
                "nome_modelo": "lasso",
                "melhores_parametros": {"alpha": 1.0},
                "melhor_score": 287603.58,
            }
        ),
    )

    chaves_metricas = [m[0] for m in metricas_logadas]
    # As métricas padronizadas rmse_cv e rmse devem ter sido logadas na run do modelo
    assert "rmse_cv" in chaves_metricas
    assert "rmse" in chaves_metricas
    # NÃO deve conter rmse_cv_lasso ou gridsearch_lasso_best_rmse
    assert "rmse_cv_lasso" not in chaves_metricas
    assert "gridsearch_lasso_best_rmse" not in chaves_metricas

    # Verifica o valor logado para rmse_cv
    valor_rmse_cv = next(v for k, v in metricas_logadas if k == "rmse_cv")
    assert valor_rmse_cv == 287603.58


def test_observador_mlflow_atualiza_repeated_kfold_na_run_filha(monkeypatch) -> None:
    import mlflow
    from mlflow.tracking import MlflowClient
    from imobiliaria_ml.mlflow_pkg.observador_mlflow import ObservadorMlflow
    import pandas as pd

    runs_filhas_metricas: dict[str, list[tuple[str, float]]] = {}

    class MockClient:
        def __init__(self, tracking_uri=None) -> None:
            pass

        def log_metric(self, run_id: str, key: str, value: float) -> None:
            if run_id not in runs_filhas_metricas:
                runs_filhas_metricas[run_id] = []
            runs_filhas_metricas[run_id].append((key, value))

    class MockRun:
        def __init__(self, run_id: str) -> None:
            self.info = type("MockInfo", (), {"run_id": run_id})()

    monkeypatch.setattr(ObservadorMlflow, "_configurar_mlflow", lambda self: None)
    monkeypatch.setattr(mlflow, "start_run", lambda run_name=None, nested=False: MockRun(f"rid_{run_name}"))
    monkeypatch.setattr(mlflow, "end_run", lambda: None)
    monkeypatch.setattr(mlflow, "active_run", lambda: MockRun("pai"))
    monkeypatch.setattr(mlflow, "set_tags", lambda t: None)
    monkeypatch.setattr(mlflow, "set_tag", lambda k, v: None)
    monkeypatch.setattr(mlflow, "log_param", lambda k, v: None)
    monkeypatch.setattr(mlflow, "log_dict", lambda **kwargs: None)
    monkeypatch.setattr(mlflow, "log_table", lambda **kwargs: None)
    monkeypatch.setattr("imobiliaria_ml.mlflow_pkg.observador_mlflow.MlflowClient", MockClient)

    obs = ObservadorMlflow(tracking_uri="http://mock-uri:5000")

    # Inicia e finaliza dois modelos candidatos
    obs.atualizar(TipoEvento.GRIDSEARCH_INICIADO, CargaEvento(valores={"nome_modelo": "random_forest"}))
    obs.atualizar(TipoEvento.GRIDSEARCH_FINALIZADO, CargaEvento(valores={"nome_modelo": "random_forest", "melhor_score": 134110.0}))

    obs.atualizar(TipoEvento.GRIDSEARCH_INICIADO, CargaEvento(valores={"nome_modelo": "xgboost"}))
    obs.atualizar(TipoEvento.GRIDSEARCH_FINALIZADO, CargaEvento(valores={"nome_modelo": "xgboost", "melhor_score": 118000.0}))

    # Emite evento de validação finalizada com resumo de modelos
    df_resumo = pd.DataFrame([
        {"modelo": "random_forest", "rmse_medio": 133000.0, "rmse_desvio": 5000.0, "mae_medio": 90000.0, "r2_medio": 0.85, "tempo_treino_medio": 2.5},
        {"modelo": "xgboost", "rmse_medio": 117000.0, "rmse_desvio": 4000.0, "mae_medio": 80000.0, "r2_medio": 0.88, "tempo_treino_medio": 1.2},
    ])

    obs.atualizar(TipoEvento.VALIDACAO_FINALIZADA, CargaEvento(valores={"resumo_modelos": df_resumo}))

    # Verifica se as métricas foram gravadas na run filha de cada modelo correspondente
    rid_rf = "rid_random_forest"
    rid_xgb = "rid_xgboost"
    assert rid_rf in runs_filhas_metricas
    assert rid_xgb in runs_filhas_metricas

    metricas_rf = dict(runs_filhas_metricas[rid_rf])
    assert metricas_rf["rmse_repeated_kfold"] == 133000.0
    assert metricas_rf["mae"] == 90000.0
    assert metricas_rf["r2"] == 0.85

    metricas_xgb = dict(runs_filhas_metricas[rid_xgb])
    assert metricas_xgb["rmse_repeated_kfold"] == 117000.0
    assert metricas_xgb["mae"] == 80000.0
    assert metricas_xgb["r2"] == 0.88


def test_observador_mlflow_com_usar_votacao_adiciona_sufixo_nas_runs(monkeypatch) -> None:
    import mlflow
    from imobiliaria_ml.mlflow_pkg.observador_mlflow import ObservadorMlflow

    runs_iniciadas: list[str] = []
    tags_logadas: list[dict[str, str]] = []

    class MockRunInfo:
        def __init__(self, run_id: str) -> None:
            self.run_id = run_id

    class MockRun:
        def __init__(self, run_id: str) -> None:
            self.info = MockRunInfo(run_id)

    run_counter = [0]

    def mock_start_run(run_name=None, nested=False):
        run_counter[0] += 1
        rid = f"run_{run_counter[0]}"
        runs_iniciadas.append(str(run_name))
        return MockRun(rid)

    active_run_holder = [None]

    def mock_active_run():
        return active_run_holder[0]

    class MockClient:
        def __init__(self, tracking_uri=None) -> None:
            pass

        def set_tag(self, run_id: str, key: str, value: str) -> None:
            pass

        def log_metric(self, run_id: str, key: str, value: float) -> None:
            pass

    monkeypatch.setattr(ObservadorMlflow, "_configurar_mlflow", lambda self: None)
    monkeypatch.setattr("imobiliaria_ml.mlflow_pkg.observador_mlflow.MlflowClient", MockClient)
    monkeypatch.setattr(mlflow, "start_run", mock_start_run)
    monkeypatch.setattr(mlflow, "end_run", lambda: None)
    monkeypatch.setattr(mlflow, "active_run", mock_active_run)
    monkeypatch.setattr(mlflow, "log_metric", lambda k, v: None)
    monkeypatch.setattr(mlflow, "set_tags", lambda t: tags_logadas.append(t))
    monkeypatch.setattr(mlflow, "set_tag", lambda k, v: None)
    monkeypatch.setattr(mlflow, "log_param", lambda k, v: None)
    monkeypatch.setattr(mlflow, "log_dict", lambda **kwargs: None)
    monkeypatch.setattr(mlflow, "log_table", lambda **kwargs: None)

    obs = ObservadorMlflow(tracking_uri="http://mock-uri:5000", usar_votacao=True)

    # 1. Run pai inicializada
    obs.atualizar(TipoEvento.EDA_FINALIZADA, CargaEvento(valores={"teste": 1}))
    assert "pipeline_treinamento_votacao" in runs_iniciadas

    # Simula active_run ativo para as etapas seguintes
    active_run_holder[0] = MockRun("pai_1")

    # 2. Run filha de candidato no GridSearch
    obs.atualizar(
        TipoEvento.GRIDSEARCH_INICIADO,
        CargaEvento(valores={"nome_modelo": "ridge"}),
    )
    assert "ridge_votacao" in runs_iniciadas

    # 3. Run filha de Ensemble
    obs.atualizar(
        TipoEvento.ENSEMBLE_FINALIZADO,
        CargaEvento(valores={"usado": True, "estrategia": "voting_ponderado_rmse"}),
    )
    assert "ensemble_voting_ponderado_rmse_votacao" in runs_iniciadas

    # 4. Tags do Modelo Campeão
    obs.atualizar(
        TipoEvento.MODELO_CAMPEAO,
        CargaEvento(valores={"nome_campeao": "ridge"}),
    )
    tags_campeao = next(t for t in tags_logadas if t.get("modelo_campeao") == "ridge")
    assert tags_campeao["mlflow.runName"] == "campeao_ridge_votacao"
    assert tags_campeao["usar_votacao"] == "true"


def test_observador_mlflow_inclui_todos_resultados_de_negocio(monkeypatch) -> None:
    """Valida o registro integral de métricas, tabelas, figuras e artefatos de negócio no MLflow."""
    import mlflow
    from imobiliaria_ml.mlflow_pkg.observador_mlflow import ObservadorMlflow
    import pandas as pd

    metricas_pai: list[tuple[str, float]] = []
    metricas_filhas: dict[str, list[tuple[str, float]]] = {}
    tags_logadas: list[dict[str, str]] = []
    dicts_logados: list[tuple[str, dict]] = []
    tabelas_logadas: list[tuple[str, pd.DataFrame]] = []
    textos_logados: list[tuple[str, str]] = []
    figuras_logadas: list[str] = []

    class MockRunInfo:
        run_id = "run_pai_id"

    class MockRun:
        info = MockRunInfo()

    class MockClient:
        def __init__(self, tracking_uri=None) -> None:
            pass

        def set_tag(self, run_id: str, key: str, value: str) -> None:
            pass

        def log_metric(self, run_id: str, key: str, value: float) -> None:
            if run_id not in metricas_filhas:
                metricas_filhas[run_id] = []
            metricas_filhas[run_id].append((key, value))

    monkeypatch.setattr(ObservadorMlflow, "_configurar_mlflow", lambda self: None)
    monkeypatch.setattr("imobiliaria_ml.mlflow_pkg.observador_mlflow.MlflowClient", MockClient)
    monkeypatch.setattr(mlflow, "start_run", lambda **kw: MockRun())
    monkeypatch.setattr(mlflow, "end_run", lambda: None)
    monkeypatch.setattr(mlflow, "active_run", lambda: MockRun())
    monkeypatch.setattr(mlflow, "log_metric", lambda k, v: metricas_pai.append((k, v)))
    monkeypatch.setattr(mlflow, "set_tags", lambda t: tags_logadas.append(t))
    monkeypatch.setattr(mlflow, "set_tag", lambda k, v: None)
    monkeypatch.setattr(mlflow, "log_param", lambda k, v: None)
    monkeypatch.setattr(mlflow, "log_dict", lambda dictionary, artifact_file: dicts_logados.append((artifact_file, dictionary)))
    monkeypatch.setattr(mlflow, "log_table", lambda data, artifact_file: tabelas_logadas.append((artifact_file, data)))
    monkeypatch.setattr(mlflow, "log_text", lambda text, artifact_file: textos_logados.append((artifact_file, text)))
    monkeypatch.setattr(mlflow, "log_figure", lambda figure, artifact_file: figuras_logadas.append(artifact_file))

    obs = ObservadorMlflow(tracking_uri="http://mock-uri:5000")
    obs._runs_modelos["ridge"] = "rid_ridge_123"
    obs._nome_campeao = "ridge"

    df_faixas = pd.DataFrame([{"faixa_preco": "Até R$ 300k", "quantidade": 50, "mae": 15000.0}])
    df_zonas = pd.DataFrame([{"Zona": "Centro", "quantidade": 100, "mae": 12000.0}])
    df_cobertura = pd.DataFrame([{"faixa_tolerancia": "±5%", "cobertura_percentual": 82.5}])

    payload_negocio = {
        "nome_campeao": "ridge",
        "metricas_consolidadas": {
            "cobertura_5": 0.825,
            "cobertura_10": 0.950,
            "cobertura_15": 0.985,
            "mae_reais": 12450.0,
            "vies_medio": 320.0,
            "desconto_seguro": 6.5,
            "risco_subprecificacao": 0.015,
            "risco_superprecificacao": 0.010,
            "margem_negociacao_estimada": 35000.0,
            "receita_potencial_perdida": 18000.0,
        },
        "tabela_erro_faixa": df_faixas,
        "tabela_erro_zona": df_zonas,
        "tabela_cobertura_tolerancia": df_cobertura,
    }

    obs.atualizar(TipoEvento.METRICAS_NEGOCIO_FINALIZADAS, CargaEvento(valores=payload_negocio))

    # 1. Verifica métricas na run pai
    chaves_metricas_pai = [m[0] for m in metricas_pai]
    assert "negocio_cobertura_5" in chaves_metricas_pai
    assert "negocio_cobertura_10" in chaves_metricas_pai
    assert "negocio_cobertura_15" in chaves_metricas_pai
    assert "negocio_mae_reais" in chaves_metricas_pai
    assert "negocio_vies_medio" in chaves_metricas_pai
    assert "negocio_desconto_seguro" in chaves_metricas_pai
    assert "negocio_risco_subprecificacao" in chaves_metricas_pai
    assert "negocio_risco_superprecificacao" in chaves_metricas_pai
    assert "negocio_margem_negociacao_estimada" in chaves_metricas_pai
    assert "negocio_receita_potencial_perdida" in chaves_metricas_pai

    # 2. Verifica métricas na linha do modelo campeão (run filha)
    assert "rid_ridge_123" in metricas_filhas
    chaves_filha = [m[0] for m in metricas_filhas["rid_ridge_123"]]
    assert "negocio_desconto_seguro" in chaves_filha
    assert "desconto_seguro" in chaves_filha
    assert "cobertura_10" in chaves_filha

    # 3. Verifica dicionários JSON logados
    arquivos_dict = [d[0] for d in dicts_logados]
    assert "negocio/metricas_imobiliaria.json" in arquivos_dict
    assert "negocio/resumo_executivo_negocio.json" in arquivos_dict

    # 4. Verifica tabelas JSON logadas via log_table
    arquivos_tab = [t[0] for t in tabelas_logadas]
    assert "negocio/metricas_por_faixa_preco.json" in arquivos_tab
    assert "negocio/metricas_por_zona.json" in arquivos_tab
    assert "negocio/cobertura_tolerancia.json" in arquivos_tab

    # 5. Verifica arquivos CSV logados via log_text
    arquivos_csv = [tx[0] for tx in textos_logados]
    assert "negocio/metricas_por_faixa_preco.csv" in arquivos_csv
    assert "negocio/metricas_por_zona.csv" in arquivos_csv
    assert "negocio/cobertura_tolerancia.csv" in arquivos_csv

    # 6. Verifica figura do gráfico de cobertura
    assert "negocio/grafico_cobertura_tolerancia.png" in figuras_logadas


def test_observador_mlflow_salva_equacao_txt_candidato_e_campeao(monkeypatch) -> None:
    import mlflow
    from mlflow.tracking import MlflowClient
    from imobiliaria_ml.mlflow_pkg.observador_mlflow import ObservadorMlflow

    textos_logados_mlflow: list[tuple[str, str]] = []
    textos_logados_client: list[tuple[str, str, str]] = []

    class MockRunInfo:
        def __init__(self, run_id: str) -> None:
            self.run_id = run_id

    class MockRun:
        def __init__(self, run_id: str) -> None:
            self.info = MockRunInfo(run_id)

    class MockClient:
        def __init__(self, tracking_uri=None) -> None:
            pass

        def log_text(self, run_id: str, text: str, artifact_file: str) -> None:
            textos_logados_client.append((run_id, text, artifact_file))

        def set_tag(self, run_id: str, key: str, value: str) -> None:
            pass

        def log_metric(self, run_id: str, key: str, value: float) -> None:
            pass

    monkeypatch.setattr(ObservadorMlflow, "_configurar_mlflow", lambda self: None)
    monkeypatch.setattr(mlflow, "start_run", lambda run_name=None, nested=False: MockRun("rid_teste"))
    monkeypatch.setattr(mlflow, "end_run", lambda: None)
    monkeypatch.setattr(mlflow, "active_run", lambda: MockRun("rid_pai"))
    monkeypatch.setattr(mlflow, "log_text", lambda text, artifact_file: textos_logados_mlflow.append((artifact_file, text)))
    monkeypatch.setattr(mlflow, "set_tag", lambda k, v: None)
    monkeypatch.setattr(mlflow, "set_tags", lambda t: None)
    monkeypatch.setattr(mlflow, "log_param", lambda k, v: None)
    monkeypatch.setattr(mlflow, "log_metric", lambda k, v: None)
    monkeypatch.setattr(mlflow, "log_dict", lambda **kwargs: None)
    monkeypatch.setattr(mlflow, "log_table", lambda **kwargs: None)
    monkeypatch.setattr("imobiliaria_ml.mlflow_pkg.observador_mlflow.MlflowClient", MockClient)

    obs = ObservadorMlflow(tracking_uri="http://mock-uri:5000")

    # 1. GridSearch de um candidato com equação no payload
    obs.atualizar(TipoEvento.GRIDSEARCH_INICIADO, CargaEvento(valores={"nome_modelo": "ridge"}))
    obs.atualizar(
        TipoEvento.GRIDSEARCH_FINALIZADO,
        CargaEvento(
            valores={
                "nome_modelo": "ridge",
                "melhores_parametros": {"alpha": 1.0},
                "melhor_score": 250000.0,
                "equacao_texto": "Valor_da_Venda = 1000 + (50 * Metragem)",
            }
        ),
    )

    arquivos_txt_mlflow = [t[0] for t in textos_logados_mlflow]
    assert "equacao_reta.txt" in arquivos_txt_mlflow
    assert "equacao_modelo.txt" in arquivos_txt_mlflow
    assert "gridsearch/ridge_equacao.txt" in arquivos_txt_mlflow

    # 2. Explicabilidade do Campeão
    obs.atualizar(
        TipoEvento.EXPLICABILIDADE_GERADA,
        CargaEvento(
            valores={
                "nome_campeao": "ridge",
                "tipo": "coeficientes",
                "equacao_texto": "Valor_da_Venda = 1000 + (50 * Metragem)",
            }
        ),
    )

    arquivos_txt_campeao = [t[0] for t in textos_logados_mlflow]
    assert "explicabilidade/equacao_campeao.txt" in arquivos_txt_campeao
    assert "explicabilidade/equacao_reta.txt" in arquivos_txt_campeao
    assert "explicabilidade/todas_equacoes_modelos.txt" in arquivos_txt_campeao





