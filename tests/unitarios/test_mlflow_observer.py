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
