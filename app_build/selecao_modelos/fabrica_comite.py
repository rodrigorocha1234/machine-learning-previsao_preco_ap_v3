from collections.abc import Callable, Mapping
from typing import Final

from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingRegressor

from app_build.selecao_modelos.contrato_seletor import DecisaoSelecao


class FabricaComite:
    @staticmethod
    def criar(
        decisao: DecisaoSelecao,
        estimadores: list[tuple[str, BaseEstimator]],
    ) -> BaseEstimator:
        criadores: Final[Mapping[str, Callable[[], BaseEstimator]]] = {
            "MODELO_UNICO": lambda: dict(estimadores)[decisao.modelo_principal],
            "ENSEMBLE_VOTACAO": lambda: VotingRegressor(estimators=estimadores),
        }
        return criadores[decisao.modo_selecao]()
