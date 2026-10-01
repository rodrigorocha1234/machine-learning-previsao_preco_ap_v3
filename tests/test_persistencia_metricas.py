from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
from prometheus_client import CollectorRegistry, Gauge

from app_build.observabilidade_metricas.persistencia_metricas import (
    PersistenciaMetricas,
)
from scripts.exportar_metricas import ExportadorMetricas


def test_snapshot_persiste_e_exporta_sem_processo_de_treino(tmp_path):
    caminho = tmp_path / "treino.prom"
    registry = CollectorRegistry()
    Gauge(
        "apartamentos_cv_desvio_padrao",
        "Teste",
        ["modelo", "metrica"],
        registry=registry,
    ).labels("ridge", "rmse").set(123)
    persistencia = PersistenciaMetricas(caminho, registry)
    persistencia.salvar()
    del persistencia, registry

    class Handler(ExportadorMetricas):
        pass

    Handler.caminho = caminho
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as servidor:
        thread = Thread(target=servidor.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{servidor.server_port}/metrics"
            with urlopen(url) as response:
                texto = response.read().decode()
            assert "123.0" in texto
            assert "apartamentos_snapshot_timestamp" in texto
            caminho.unlink()
            with pytest.raises(HTTPError) as falha:
                urlopen(url)
            assert falha.value.code == 503
        finally:
            servidor.shutdown()
            thread.join()


def test_falha_na_coleta_preserva_snapshot_anterior(tmp_path):
    caminho = tmp_path / "treino.prom"
    registry = CollectorRegistry()
    Gauge("teste", "Teste", registry=registry).set(7)
    PersistenciaMetricas(caminho, registry).salvar()
    anterior = caminho.read_bytes()

    class ColetorFalho:
        def collect(self):
            raise ValueError("Falha de coleta")

    registry.register(ColetorFalho())
    with pytest.raises(ValueError):
        PersistenciaMetricas(caminho, registry).salvar()
    assert caminho.read_bytes() == anterior
    assert not list(tmp_path.glob(".treino-*"))
