from contextlib import nullcontext
from dataclasses import asdict
from functools import partial

import numpy as np
import pandas as pd
import pytest
from prometheus_client import CollectorRegistry

from app_build.validacao_cruzada.acumulador_metricas import AcumuladorMetricas
from app_build.validacao_cruzada.contrato_validador import MetricasRegressao


def test_desvios_conhecidos_para_todas_as_metricas():
    metricas = tuple(
        MetricasRegressao(v, 2 * v, 3 * v, 4 * v, 5 * v, 6 * v) for v in (1.0, 2.0, 3.0)
    )
    resultado = AcumuladorMetricas.agregar_desvios_padrao(metricas)
    esperado = np.sqrt(2 / 3)
    assert list(asdict(resultado).values()) == pytest.approx(
        [esperado * i for i in range(1, 7)]
    )


def test_folds_identicos_e_fold_unico_tem_desvio_zero():
    metrica = MetricasRegressao(10.0, 5.0, 100.0, 0.9, 0.1, 0.05)
    for folds in ((metrica,), (metrica, metrica, metrica)):
        assert list(
            asdict(AcumuladorMetricas.agregar_desvios_padrao(folds)).values()
        ) == pytest.approx([0.0] * 6)
    with pytest.raises(ValueError):
        AcumuladorMetricas.agregar_desvios_padrao(())


def test_nested_cv_registra_dispersao_no_mlflow_e_prometheus(monkeypatch):
    import mlflow

    from app_build.observabilidade_metricas import coletor_prometheus as modulo_coletor
    from app_build.rastreamento_mlflow.contrato_observador import (
        EventoNestedCvConcluido,
    )
    from app_build.rastreamento_mlflow.observador_mlflow import ObservadorMlflow
    from app_build.validacao_cruzada.avaliador_aninhado import AvaliadorAninhado
    from app_build.validacao_cruzada.particionador_externo import ParticionadorExterno

    x = pd.DataFrame(
        {
            "Metragem": np.arange(30.0, 90.0, 2.0),
            "Quartos": np.tile([1, 2, 3], 10),
            "Banheiros": np.tile([1, 2], 15),
            "Vagas_Garagem": np.tile([0, 1, 2], 10),
            "Zona": ["Sul"] * 30,
            "Bairro": ["Jardim"] * 30,
        }
    )
    y = x["Metragem"] * 5000 + np.random.default_rng(42).normal(0, 10000, 30)
    divisoes = ParticionadorExterno(splits=3, repeticoes=1, semente=42).gerar_divisoes(
        x
    )
    resultado = AvaliadorAninhado(
        {"tuning": {"estrategia": "nenhum"}}, divisoes
    ).avaliar_modelo("linear_regression", x, y)
    assert len(resultado.resultados_folds) == 3
    for nome, desvio in asdict(resultado.metricas_desvios_padrao).items():
        assert desvio == pytest.approx(
            np.std(
                [getattr(f.metricas, nome) for f in resultado.resultados_folds], ddof=0
            )
        )

    logs = {}
    params = {}
    monkeypatch.setattr(mlflow, "start_run", lambda **kwargs: nullcontext())
    monkeypatch.setattr(mlflow, "set_tag", lambda *args: None)
    monkeypatch.setattr(mlflow, "log_metric", lambda *args: None)
    monkeypatch.setattr(mlflow, "log_metrics", logs.update)
    monkeypatch.setattr(
        mlflow, "log_param", lambda nome, valor: params.update({nome: valor})
    )
    monkeypatch.setattr(mlflow, "log_artifact", lambda *args, **kwargs: None)
    ObservadorMlflow().ao_concluir_nested_cv(
        EventoNestedCvConcluido(
            nome_modelo="linear_regression",
            resultado_cv=resultado,
            explicacoes_parametros=(),
        )
    )
    assert params["desvio_padrao_ddof"] == 0
    assert params["total_folds_externos"] == 3

    registry = CollectorRegistry()
    for tipo in ("Gauge", "Counter", "Histogram", "Info"):
        monkeypatch.setattr(
            modulo_coletor,
            tipo,
            partial(getattr(modulo_coletor, tipo), registry=registry),
        )
    coletor = modulo_coletor.ColetorPrometheus()
    coletor.registrar_dispersao_cv(resultado)
    for nome, desvio in asdict(resultado.metricas_desvios_padrao).items():
        assert logs[f"{nome}_std"] == pytest.approx(desvio)
        assert registry.get_sample_value(
            "apartamentos_cv_desvio_padrao",
            {"modelo": "linear_regression", "metrica": nome},
        ) == pytest.approx(desvio)
        assert registry.get_sample_value(
            "apartamentos_cv_media", {"modelo": "linear_regression", "metrica": nome}
        ) == pytest.approx(getattr(resultado.metricas_medias, nome))
