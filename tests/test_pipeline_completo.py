from pathlib import Path

import pandas as pd
import pytest

from app_build.ajuste_modelos.fabrica_estimadores import FabricaEstimadores
from app_build.camada_dados.validador_contrato import (
    ValidadorContrato,
    ViolacaoContratoDadosErro,
)
from app_build.configuracao_sistema.armazem_configuracao import ArmazemConfiguracao
from app_build.estatistica_modelos.teste_friedman import TesteFriedman
from app_build.estatistica_modelos.teste_nemenyi import TesteNemenyi
from app_build.isolamento_dados.cofre_holdout import (
    CofreHoldout,
    ViolacaoIsolamentoDadosErro,
)
from app_build.isolamento_dados.divisor_estratificado import DivisorEstratificado
from app_build.observabilidade_metricas.coletor_prometheus import ColetorPrometheus
from app_build.processamento_dados.construtor_pipeline import ConstrutorPipeline
from app_build.regras_negocio.agregador_hierarquico import AgregadorHierarquico
from app_build.regras_negocio.motor_imobiliario import MotorImobiliario


@pytest.fixture
def base_exemplo() -> pd.DataFrame:
    dados = {
        "Código": [101, 102, 103, 104, 105],
        "Apartamento": ["Ap 1", "Ap 2", "Ap 3", "Ap 4", "Ap 5"],
        "Bairro": ["Jardim Botânico", "Jardim Botânico", "Centro", "Centro", "Nova Aliança"],
        "Zona": ["Zona Sul", "Zona Sul", "Centro", "Centro", "Zona Sul"],
        "Quartos": [2, 3, 1, 2, 4],
        "Banheiros": [2, 3, 1, 1, 4],
        "Vagas_Garagem": [1, 2, 0, 1, 3],
        "Metragem": [65.0, 110.0, 45.0, 55.0, 160.0],
        "Valor_da_Venda": [350000.0, 750000.0, 200000.0, 260000.0, 1200000.0],
    }
    return pd.DataFrame(dados)


def test_armazem_configuracao():
    armazem = ArmazemConfiguracao()
    armazem.inicializar()
    assert armazem.geral.dados.target == "Valor_da_Venda"
    assert "random_forest" in armazem.modelos
    assert armazem.modelos["random_forest"]["parametros"]["n_jobs"] == 3


def test_validador_contrato_sucesso(base_exemplo):
    validador = ValidadorContrato("Valor_da_Venda")
    validador.validar(base_exemplo)


def test_validador_contrato_falha():
    validador = ValidadorContrato("Valor_da_Venda")
    df_invalido = pd.DataFrame({"Valor_da_Venda": [-100.0]})
    with pytest.raises(ViolacaoContratoDadosErro):
        validador.validar(df_invalido)


def test_isolamento_holdout(base_exemplo):
    divisor = DivisorEstratificado(proporcao_holdout=0.20, semente=42)
    particao = divisor.dividir(base_exemplo)
    cofre = CofreHoldout()
    cofre.bloquear(particao.holdout)

    with pytest.raises(ViolacaoIsolamentoDadosErro):
        cofre.liberar_holdout("CHAVE_INCORRETA")

    holdout_aberto = cofre.liberar_holdout(CofreHoldout.CHAVE_MESTRA)
    assert len(holdout_aberto) == len(particao.holdout)

    # Segundo acesso bloqueado
    with pytest.raises(ViolacaoIsolamentoDadosErro):
        cofre.liberar_holdout(CofreHoldout.CHAVE_MESTRA)


def test_pipeline_preprocessamento(base_exemplo):
    dados_x = base_exemplo.drop(columns=["Valor_da_Venda"])
    vetor_y = base_exemplo["Valor_da_Venda"]
    construtor = ConstrutorPipeline()
    pipe = construtor.criar_preprocessador()
    matriz = pipe.fit_transform(dados_x, vetor_y)
    assert matriz.shape[0] == len(dados_x)
    assert matriz.shape[1] > 5


def test_fabrica_10_estimadores():
    modelos = [
        "linear_regression",
        "ridge",
        "lasso",
        "elastic_net",
        "arvore_decisao",
        "random_forest",
        "svr",
        "rede_neural",
        "xgboost",
        "lightgbm",
    ]
    for mod in modelos:
        est = FabricaEstimadores.criar_estimador(mod)
        assert hasattr(est, "fit")
        assert hasattr(est, "predict")


def test_regras_negocio_hierarquia_global_zona_bairro(base_exemplo):
    agregador = AgregadorHierarquico(minima_zona=2, minima_bairro=2)
    estatisticas = agregador.calcular_estatisticas(base_exemplo, "Valor_da_Venda")
    motor = MotorImobiliario(estatisticas)

    resultado = motor.enriquecer_previsao(
        valor_previsto=500000.0,
        metragem=80.0,
        zona="Zona Sul",
        bairro="Jardim Botânico",
    )

    # Validação da hierarquia GLOBAL -> ZONA -> BAIRRO
    assert resultado.estatisticas_global.nivel == "GLOBAL"
    assert resultado.estatisticas_zona.nivel == "ZONA"
    assert resultado.estatisticas_bairro.nivel == "BAIRRO"
    assert resultado.valor_previsto == 500000.0
    assert resultado.valor_m2_previsto == 500000.0 / 80.0
    assert resultado.desconto_moderado_5 == 500000.0 * 0.95
    assert resultado.faixa_segura_piso < resultado.faixa_segura_teto


def test_empacotador_calcula_previsoes_por_zona_e_bairro(base_exemplo):
    from sklearn.base import BaseEstimator

    from app_build.rastreamento_mlflow.empacotador_modelo import EmpacotadorModelo

    class EstimadorTeste(BaseEstimator):
        def predict(self, dados: pd.DataFrame) -> list[float]:
            return [100000.0, 300000.0, 500000.0]

    estatisticas = AgregadorHierarquico(
        minima_zona=1, minima_bairro=1
    ).calcular_estatisticas(base_exemplo, "Valor_da_Venda")
    empacotador = EmpacotadorModelo(
        pipeline_scikit=EstimadorTeste(),
        motor_imobiliario=MotorImobiliario(estatisticas),
    )
    dados = base_exemplo.iloc[[0, 1, 4]].drop(columns=["Valor_da_Venda"])

    previsoes = empacotador.predict(context=None, model_input=dados)

    assert previsoes["valor_previsto_zona"].tolist() == pytest.approx(
        [300000.0, 300000.0, 300000.0]
    )
    assert previsoes["valor_previsto_bairro"].tolist() == pytest.approx(
        [200000.0, 200000.0, 500000.0]
    )
    assert previsoes["valor_previsto_medio_zona"].tolist() == pytest.approx(
        previsoes["valor_previsto_zona"].tolist()
    )
    assert previsoes["valor_previsto_medio_bairro"].tolist() == pytest.approx(
        previsoes["valor_previsto_bairro"].tolist()
    )
    assert previsoes["valor_m2_previsto_zona"].tolist() == pytest.approx(
        [
            (100000.0 / 65.0 + 300000.0 / 110.0 + 500000.0 / 160.0) / 3,
        ]
        * 3
    )
    assert previsoes["valor_m2_previsto_bairro"].tolist() == pytest.approx(
        [(100000.0 / 65.0 + 300000.0 / 110.0) / 2] * 2
        + [500000.0 / 160.0]
    )


def test_estatisticas_friedman_nemenyi():
    scores = pd.DataFrame({
        "mod_a": [10.0, 11.0, 12.0, 10.5, 11.2],
        "mod_b": [20.0, 21.0, 22.0, 20.5, 21.2],
        "mod_c": [15.0, 16.0, 17.0, 15.5, 16.2],
    })
    tf = TesteFriedman()
    res_f = tf.testar(scores)
    assert res_f.eh_significativo

    tn = TesteNemenyi()
    res_n = tn.testar(res_f, total_folds=5)
    assert res_n.diferenca_critica > 0.0
    assert len(res_n.comparacoes) == 3


def test_telemetria_prometheus():
    coletor = ColetorPrometheus()
    coletor.registrar_inferencia(0.015, "Zona Sul", 850000.0)
    coletor.atualizar_metricas_modelo(rmse=25000.0, mae=18000.0, r2=0.88)
    coletor.registrar_holdout_zona(
        zona="Zona Sul", rmse=25000.0, mae=18000.0, r2=0.88,
        mape=4.2, total_amostras=12, valor_medio_previsto=720000.0,
    )
    coletor.registrar_holdout_bairro(
        bairro="Jardim Botânico", zona="Zona Sul", rmse=22000.0, r2=0.9,
        valor_medio_previsto=810000.0,
    )
    assert coletor._gauge_holdout_valor_medio_previsto_zona.labels(
        zona="Zona Sul"
    )._value.get() == 720000.0
    assert coletor._gauge_holdout_valor_medio_previsto_bairro.labels(
        bairro="Jardim Botânico", zona="Zona Sul"
    )._value.get() == 810000.0


def test_regras_negocio_rastreamento_mlflow(
    base_exemplo: pd.DataFrame, tmp_path: Path
) -> None:
    from sklearn.linear_model import Ridge

    from app_build.rastreamento_mlflow.contrato_observador import (
        EventoRegrasNegocioConcluido,
        EventoTreinoFinalConcluido,
    )
    from app_build.rastreamento_mlflow.observador_mlflow import ObservadorMlflow

    agregador = AgregadorHierarquico(minima_zona=1, minima_bairro=1)
    estatisticas = agregador.calcular_estatisticas(base_exemplo)
    motor = MotorImobiliario(estatisticas)
    df_enriquecido = motor.enriquecer_dataframe(
        base_exemplo, base_exemplo["Valor_da_Venda"]
    )

    observador = ObservadorMlflow(
        nome_experimento="teste_regras_negocio",
        tracking_uri=f"file://{tmp_path}",
    )
    observador.inicializar()

    exemplo_res = motor.enriquecer_previsao(
        350000.0, 65.0, "Zona Sul", "Jardim Botânico"
    )
    evento_regras = EventoRegrasNegocioConcluido(
        estatisticas_hierarquicas=estatisticas,
        dados_enriquecidos=df_enriquecido,
        exemplos_simulacao=(exemplo_res,),
        metricas_negocio={"desconto_5_medio": 332500.0},
    )
    observador.ao_concluir_regras_negocio(evento_regras)

    modelo = Ridge()
    modelo.fit(base_exemplo[["Metragem", "Quartos"]], base_exemplo["Valor_da_Venda"])
    evento_treino = EventoTreinoFinalConcluido(
        nome_modelo="ridge",
        estimador=modelo,
        explicacoes_parametros=(),
        dados_exemplo=base_exemplo[["Metragem", "Quartos"]],
    )
    observador.ao_concluir_treino_final(evento_treino)


def test_executor_esteira_logs_etapas() -> None:
    from app_build.observabilidade_metricas.emissor_logs import EmissorLogs, EventoLog
    from app_build.orquestracao_pipeline.contexto_execucao import ContextoExecucao
    from app_build.orquestracao_pipeline.contrato_etapa import ContratoEtapa
    from app_build.orquestracao_pipeline.executor_esteira import ExecutorEsteira

    logs_capturados: list[EventoLog] = []

    class MockEmissor(EmissorLogs):
        def emitir_evento(self, evento: EventoLog) -> bool:
            logs_capturados.append(evento)
            return True

    class EtapaTeste1(ContratoEtapa):
        @property
        def nome_etapa(self) -> str:
            return "01_teste_alpha"

        def executar(self, contexto: ContextoExecucao) -> None:
            contexto.estado_congelado = True

    class EtapaTeste2(ContratoEtapa):
        @property
        def nome_etapa(self) -> str:
            return "02_teste_beta"

        def executar(self, contexto: ContextoExecucao) -> None:
            pass

    executor = ExecutorEsteira(
        etapas=(EtapaTeste1(), EtapaTeste2()),
        emissor_logs=MockEmissor(),
    )
    ctx = executor.executar_esteira()

    assert ctx.estado_congelado is True
    assert len(logs_capturados) == 4
    assert logs_capturados[0].mensagem == ">>> Executando etapa: 01_teste_alpha (Etapa 1 de 2)"
    assert logs_capturados[0].servico == "executor_esteira"
    assert logs_capturados[0].nivel == "INFO"
    assert logs_capturados[1].mensagem == "--- Concluida etapa: 01_teste_alpha (Etapa 1 de 2)"
    assert logs_capturados[2].mensagem == ">>> Executando etapa: 02_teste_beta (Etapa 2 de 2)"
    assert logs_capturados[3].mensagem == "--- Concluida etapa: 02_teste_beta (Etapa 2 de 2)"


def test_remocao_parametros_indesejados_api_mlflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    base_exemplo: pd.DataFrame,
) -> None:
    import mlflow
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline

    from app_build.rastreamento_mlflow.contrato_observador import (
        EventoTreinoFinalConcluido,
    )
    from app_build.rastreamento_mlflow.observador_mlflow import ObservadorMlflow

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    df_com_indesejados = pd.DataFrame({
        "Código": [101, 102],
        "Apartamento": ["Ap 1", "Ap 2"],
        "Bairro": ["Jardim Botânico", "Jardim Botânico"],
        "Zona": ["Zona Sul", "Zona Sul"],
        "Banheiros": [2, 2],
        "Vagas_Garagem": [1, 2],
        "Metragem": [60.0, 80.0],
        "Quartos": [2, 3],
        "valor_m2": [5000.0, 6000.0],
        "media_valor_m2_bairro": [5200.0, 5800.0],
        "media_valor_m2_zona": [5100.0, 5700.0],
    })

    modelo = make_pipeline(
        ColumnTransformer(
            [("numericas", "passthrough", ["Metragem", "Quartos"])]
        ),
        Ridge(),
    )
    modelo.fit(df_com_indesejados[["Metragem", "Quartos"]], [300000.0, 480000.0])
    motor = MotorImobiliario(
        AgregadorHierarquico(minima_zona=1, minima_bairro=1).calcular_estatisticas(
            base_exemplo
        )
    )

    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    observador = ObservadorMlflow(
        nome_experimento="teste_remocao_parametros",
        tracking_uri=tracking_uri,
    )
    observador.inicializar()

    evento = EventoTreinoFinalConcluido(
        nome_modelo="ridge",
        estimador=modelo,
        explicacoes_parametros=(),
        dados_exemplo=df_com_indesejados,
        motor_imobiliario=motor,
    )
    observador.ao_concluir_treino_final(evento)

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name("teste_remocao_parametros")
    assert exp is not None
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) >= 1
    run_final = runs[0]
    
    import json
    import yaml

    caminho_mlmodel = client.download_artifacts(run_final.info.run_id, "modelo/MLmodel")
    conteudo_yaml = yaml.safe_load(Path(caminho_mlmodel).read_text(encoding="utf-8"))
    inputs_meta = json.loads(conteudo_yaml["signature"]["inputs"])
    outputs_meta = json.loads(conteudo_yaml["signature"]["outputs"])

    nomes_inputs = [campo["name"] for campo in inputs_meta]
    nomes_outputs = [campo["name"] for campo in outputs_meta]

    assert "Código" not in nomes_inputs
    assert "Apartamento" not in nomes_inputs
    assert "valor_m2" not in nomes_inputs
    assert "media_valor_m2_bairro" not in nomes_inputs
    assert "media_valor_m2_zona" not in nomes_inputs

    # Validar que os resultados da regra de negócio foram acrescentados no OUTPUT
    assert "valor_previsto" in nomes_outputs
    assert "valor_m2_previsto" in nomes_outputs
    assert "valor_previsto_medio_zona" in nomes_outputs
    assert "valor_previsto_medio_bairro" in nomes_outputs
    assert "valor_previsto_zona" in nomes_outputs
    assert "valor_previsto_bairro" in nomes_outputs
    assert "valor_m2_previsto_zona" in nomes_outputs
    assert "valor_m2_previsto_bairro" in nomes_outputs
    assert "Zona" in nomes_outputs
    assert "Bairro" in nomes_outputs
    assert "indice_imovel_global" in nomes_outputs
    assert "indice_imovel_zona" in nomes_outputs
    assert "indice_imovel_bairro" in nomes_outputs
    # Nível Global
    assert "global_desconto_5" in nomes_outputs
    assert "global_desconto_10" in nomes_outputs
    assert "global_desconto_15" in nomes_outputs
    assert "global_faixa_segura_piso" in nomes_outputs
    assert "global_faixa_segura_teto" in nomes_outputs
    # Nível Zona
    assert "zona_desconto_5" in nomes_outputs
    assert "zona_desconto_10" in nomes_outputs
    assert "zona_desconto_15" in nomes_outputs
    assert "zona_faixa_segura_piso" in nomes_outputs
    assert "zona_faixa_segura_teto" in nomes_outputs
    # Nível Bairro
    assert "bairro_desconto_5" in nomes_outputs
    assert "bairro_desconto_10" in nomes_outputs
    assert "bairro_desconto_15" in nomes_outputs
    assert "bairro_faixa_segura_piso" in nomes_outputs
    assert "bairro_faixa_segura_teto" in nomes_outputs




