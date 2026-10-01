from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import Pipeline

from app_build.configuracao_sistema.armazem_configuracao import ArmazemConfiguracao
from app_build.estatistica_modelos.contrato_estatistica import ResultadoFriedman
from app_build.orquestracao_pipeline.contexto_execucao import ContextoExecucao
from app_build.orquestracao_pipeline.fabrica_etapas import (
    Etapa13TuningFinal,
    Etapa14TreinamentoFinal,
    Etapa19RastreamentoMlflow,
)
from app_build.rastreamento_mlflow.empacotador_modelo import EmpacotadorModelo
from app_build.rastreamento_mlflow.observador_mlflow import ObservadorMlflow
from app_build.selecao_modelos.seletor_campeao import SeletorCampeao


@pytest.fixture
def contexto_votacao(monkeypatch):
    armazem = ArmazemConfiguracao()
    armazem.inicializar()
    contexto = ContextoExecucao()
    contexto.configuracao_geral = replace(
        armazem.geral,
        validacao=replace(armazem.geral.validacao, splits_internos=3),
    )
    contexto.configuracao_modelos = {
        "ridge": {
            "parametros": {"alpha": 1.0},
            "tuning": {"estrategia": "grid", "parametros": {"modelo__alpha": [0.3]}},
        },
        "arvore_decisao": {
            "parametros": {"max_depth": 1, "random_state": 42},
            "tuning": {
                "estrategia": "random",
                "n_iter": 1,
                "parametros": {"modelo__max_depth": [3]},
            },
        },
        "nao_selecionado": {"tuning": {"estrategia": "invalida"}},
    }
    x = pd.DataFrame(
        {
            "Metragem": np.arange(30.0, 90.0, 2.0),
            "Quartos": np.tile([1, 2, 3], 10),
            "Banheiros": np.tile([1, 2], 15),
            "Vagas_Garagem": np.tile([0, 1, 2], 10),
            "Zona": np.tile(["Sul", "Norte"], 15),
            "Bairro": np.tile(["Jardim", "Centro"], 15),
        }
    )
    contexto.dados_desenvolvimento = x.assign(
        Valor_da_Venda=x["Metragem"] ** 2 * 100 + x["Quartos"] * 5000
    )
    resultados = {
        nome: SimpleNamespace(metricas_medianas=SimpleNamespace(rmse=rmse))
        for nome, rmse in (("ridge", 10), ("arvore_decisao", 20))
    }
    friedman = ResultadoFriedman(1.0, 0.5, False, {"arvore_decisao": 2, "ridge": 1})
    contexto.decisao_selecao = SeletorCampeao(True, 2).executar_selecao(
        resultados, friedman
    )
    monkeypatch.setattr(
        contexto.cofre_holdout,
        "liberar_holdout",
        Mock(side_effect=AssertionError("Holdout nao pode participar do ajuste")),
    )
    return contexto, resultados, friedman


@pytest.mark.parametrize(
    "usar_votacao,quantidade", [(True, 2), (True, 1), (True, 5), (False, 2)]
)
def test_selecao_tuning_e_treino_final(contexto_votacao, usar_votacao, quantidade):
    contexto, resultados, friedman = contexto_votacao
    contexto.decisao_selecao = SeletorCampeao(
        usar_votacao, quantidade
    ).executar_selecao(resultados, friedman)
    Etapa13TuningFinal().executar(contexto)
    Etapa14TreinamentoFinal().executar(contexto)
    modelo = contexto.modelo_campeao_final
    dados_x = contexto.dados_desenvolvimento.drop(columns="Valor_da_Venda")
    assert contexto.parametros_componentes["ridge"]["alpha"] == 0.3
    assert set(contexto.parametros_componentes) == set(
        contexto.decisao_selecao.modelos_selecionados
    )
    if usar_votacao:
        assert isinstance(modelo, VotingRegressor)
        assert len(modelo.estimators_) == min(quantidade, 2)
        assert all(isinstance(membro, Pipeline) for membro in modelo.estimators_)
        assert modelo.named_estimators_["ridge"].named_steps["modelo"].alpha == 0.3
        previsoes = np.asarray(
            [membro.predict(dados_x) for membro in modelo.estimators_]
        )
        np.testing.assert_allclose(modelo.predict(dados_x), previsoes.mean(axis=0))
        if quantidade > 1:
            assert (
                modelo.named_estimators_["arvore_decisao"]
                .named_steps["modelo"]
                .max_depth
                == 3
            )
            assert not np.allclose(modelo.predict(dados_x), previsoes[0])
        assert contexto.decisao_selecao.nome_modelo_final == "voting_regressor"
    else:
        assert isinstance(modelo, Pipeline)
        assert modelo.named_steps["modelo"].alpha == 0.3
        assert contexto.decisao_selecao.nome_modelo_final == "ridge"
    contexto.cofre_holdout.liberar_holdout.assert_not_called()
    assert contexto.dados_holdout_liberados is None


def test_voting_regressor_serializado_no_pyfunc_preserva_saida_api(
    contexto_votacao, tmp_path
):
    contexto, _, _ = contexto_votacao
    Etapa13TuningFinal().executar(contexto)
    Etapa14TreinamentoFinal().executar(contexto)
    modelo = EmpacotadorModelo(
        contexto.modelo_campeao_final, contexto.motor_imobiliario
    )
    dados_x = (
        contexto.dados_desenvolvimento.drop(columns="Valor_da_Venda").head(3).copy()
    )
    dados_x.loc[0, ["Zona", "Bairro"]] = ["Zona nova", "Bairro novo"]
    esperado = modelo.predict(None, dados_x)
    destino = tmp_path / "modelo"
    mlflow.pyfunc.save_model(
        path=str(destino),
        python_model=modelo,
        pip_requirements=[],
        signature=mlflow.models.infer_signature(dados_x, esperado),
    )
    restaurado = mlflow.pyfunc.load_model(str(destino))
    pd.testing.assert_frame_equal(restaurado.predict(dados_x), esperado)
    np.testing.assert_allclose(
        esperado["valor_previsto"], contexto.modelo_campeao_final.predict(dados_x)
    )
    assert len(esperado.columns) == 40
    assert {"valor_previsto_zona", "valor_previsto_bairro"} <= set(esperado.columns)


def test_mlflow_identifica_comite_e_registra_parametros(contexto_votacao, monkeypatch):
    contexto, _, _ = contexto_votacao
    Etapa13TuningFinal().executar(contexto)
    Etapa14TreinamentoFinal().executar(contexto)
    eventos = []
    monkeypatch.setattr(contexto.despachante, "despachar_treino_final", eventos.append)
    Etapa19RastreamentoMlflow().executar(contexto)
    evento = eventos[0]
    assert evento.nome_modelo == "voting_regressor"
    assert evento.parametros_componentes["arvore_decisao"]["max_depth"] == 3
    assert any(
        exp.parametro == "ridge.alpha" and exp.valor == 0.3
        for exp in evento.explicacoes_parametros
    )

    runs, tags, artefatos = [], {}, {}
    monkeypatch.setattr(
        mlflow, "start_run", lambda **kw: (runs.append(kw), nullcontext())[1]
    )
    monkeypatch.setattr(
        mlflow, "set_tag", lambda chave, valor: tags.update({chave: valor})
    )
    monkeypatch.setattr(
        mlflow, "log_dict", lambda dados, caminho: artefatos.update({caminho: dados})
    )
    monkeypatch.setattr(
        mlflow, "log_text", lambda dados, caminho: artefatos.update({caminho: dados})
    )
    registrar = Mock(return_value=SimpleNamespace(registered_model_version="2"))
    monkeypatch.setattr(mlflow.pyfunc, "log_model", registrar)
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", Mock(return_value=Mock()))
    ObservadorMlflow().ao_concluir_treino_final(evento)
    assert runs == [{"run_name": "modelo_final_voting_regressor"}]
    assert tags["modelo"] == "voting_regressor"
    assert (
        artefatos["treino_final/parametros_componentes.json"]
        == contexto.parametros_componentes
    )
    assert "ridge.alpha" in artefatos["treino_final/interpretacao_parametros.md"]
    assert isinstance(
        registrar.call_args.kwargs["python_model"].pipeline_scikit, VotingRegressor
    )
