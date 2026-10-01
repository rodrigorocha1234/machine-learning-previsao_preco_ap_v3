import asyncio
import json

import pandas as pd
import pytest
from prometheus_client import generate_latest
from prometheus_client.parser import text_string_to_metric_families

from app_build.observabilidade_metricas.adaptador_serving import AdaptadorServing
from app_build.observabilidade_metricas.metricas_serving import MetricasServing


def coletor():
    return MetricasServing({"Sul", "Centro"}, {"Jardim", "Centro"})


def test_media_ponderada_por_imovel_e_histograma():
    m = coletor()
    m.registrar_saida(
        pd.DataFrame(
            {
                "Zona": ["Sul", "Sul"],
                "Bairro": ["Jardim", "Jardim"],
                "valor_previsto": [100000.0, 300000.0],
                "valor_m2_previsto": [2000.0, 4000.0],
            }
        )
    )
    m.registrar_saida(
        pd.DataFrame(
            {
                "Zona": ["Sul"],
                "Bairro": ["Jardim"],
                "valor_previsto": [800000.0],
                "valor_m2_previsto": [8000.0],
            }
        )
    )
    labels = {"zona": "Sul", "bairro": "Jardim"}
    assert m.registry.get_sample_value("apartamentos_serving_valor_count", labels) == 3
    assert (
        m.registry.get_sample_value("apartamentos_serving_valor_sum", labels) / 3
        == 400000
    )
    assert (
        m.registry.get_sample_value(
            "apartamentos_serving_valor_bucket", {**labels, "le": "300000.0"}
        )
        == 2
    )
    families = list(
        text_string_to_metric_families(generate_latest(m.registry).decode())
    )
    histogram = next(f for f in families if f.name == "apartamentos_serving_valor")
    assert histogram.type == "histogram"
    assert next(s.value for s in histogram.samples if s.labels.get("le") == "+Inf") == 3


def test_entradas_invalidas_e_localidades_limitadas():
    m = coletor()
    m.registrar_entrada(
        pd.DataFrame(
            {
                "Zona": ["Desconhecida 1", "Sul"],
                "Bairro": ["Desconhecido 1", None],
                "Metragem": [-1, "abc"],
            }
        )
    )
    assert (
        m.registry.get_sample_value(
            "apartamentos_serving_entrada_problemas_total",
            {"campo": "Metragem", "motivo": "invalida"},
        )
        == 2
    )
    assert (
        m.registry.get_sample_value(
            "apartamentos_serving_entrada_problemas_total",
            {"campo": "Bairro", "motivo": "ausente"},
        )
        == 1
    )
    m.registrar_saida(
        pd.DataFrame(
            {
                "Zona": ["Desconhecida 1", "Sul"],
                "Bairro": ["Desconhecido 1", "Desconhecido 2"],
                "valor_previsto": [100000.0, 200000.0],
                "valor_m2_previsto": [2000.0, 4000.0],
            }
        )
    )
    assert (
        m.registry.get_sample_value(
            "apartamentos_serving_referencia_total", {"nivel": "global"}
        )
        == 1
    )
    assert (
        m.registry.get_sample_value(
            "apartamentos_serving_referencia_total", {"nivel": "zona"}
        )
        == 1
    )
    texto = generate_latest(m.registry).decode()
    assert "Desconhecido 1" not in texto
    assert "DESCONHECIDO" in texto


@pytest.mark.parametrize("status", [200, 400, 500])
def test_http_preserva_resposta_e_contabiliza_status(status):
    m = coletor()
    resposta = (
        {
            "predictions": [
                {
                    "Zona": "Sul",
                    "Bairro": "Jardim",
                    "valor_previsto": 300000.0,
                    "valor_m2_previsto": 5000.0,
                }
            ]
        }
        if status == 200
        else {"error_code": "TESTE"}
    )

    async def app(scope, receive, send):
        await receive()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {"type": "http.response.body", "body": json.dumps(resposta).encode()}
        )

    adaptador = AdaptadorServing(app, m)

    async def chamar(caminho, payload):
        mensagens = []

        async def receive():
            return {"type": "http.request", "body": json.dumps(payload).encode()}

        async def send(mensagem):
            mensagens.append(mensagem)

        await adaptador(
            {
                "type": "http",
                "path": caminho,
                "headers": [(b"content-type", b"application/json")],
            },
            receive,
            send,
        )
        return mensagens

    payload = {
        "dataframe_records": [
            {
                "Zona": "Sul",
                "Bairro": "Jardim",
                "Metragem": 60.0,
                "Quartos": 2,
                "Banheiros": 1,
                "Vagas_Garagem": 1,
            }
        ]
    }
    result = asyncio.run(chamar("/invocations", payload))
    assert result[0]["status"] == status
    assert json.loads(result[1]["body"]) == resposta
    assert (
        m.registry.get_sample_value(
            "apartamentos_serving_requisicoes_total", {"status": f"{status // 100}xx"}
        )
        == 1
    )
    assert m.registry.get_sample_value("apartamentos_serving_entradas_total") == 1
    metricas = asyncio.run(chamar("/metrics", {}))
    assert metricas[0]["status"] == 200
    assert (
        sum(
            m.registry.get_sample_value(
                "apartamentos_serving_requisicoes_total", {"status": x}
            )
            for x in ("2xx", "3xx", "4xx", "5xx")
        )
        == 1
    )


def test_json_invalido_nao_altera_resposta_e_split_e_coletado():
    m = coletor()
    adaptador = AdaptadorServing(None, m)
    adaptador._registrar_json(b"{invalido", False)
    assert (
        m.registry.get_sample_value("apartamentos_serving_telemetria_falhas_total") == 1
    )
    payload = {
        "dataframe_split": {
            "columns": [
                "Zona",
                "Bairro",
                "Metragem",
                "Quartos",
                "Banheiros",
                "Vagas_Garagem",
            ],
            "data": [["Sul", "Jardim", 60.0, 2, 1, 1]],
        }
    }
    adaptador._registrar_json(json.dumps(payload).encode(), False)
    assert m.registry.get_sample_value("apartamentos_serving_entradas_total") == 1
