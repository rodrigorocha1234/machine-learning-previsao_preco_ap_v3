"""Estratégia de VotingRegressor com pesos ponderados."""

from sklearn.ensemble import VotingRegressor
from ..modelos_pkg.regressor_protocol import RegressorProtocol
from .estrategia_ensemble import EstrategiaEnsemble


class VotacaoRegressorPonderada(EstrategiaEnsemble):
    """Ensemble por votação ponderada por métrica ou ranking."""

    def __init__(self, pesos_por_modelo: dict[str, float]) -> None:
        self._pesos_por_modelo = pesos_por_modelo

    def criar_estimador(
        self,
        estimadores_elegiveis: list[tuple[str, RegressorProtocol]],
    ) -> RegressorProtocol:
        """Cria VotingRegressor ponderado pelos pesos previamente calculados."""
        if not estimadores_elegiveis:
            raise ValueError("Ao menos um estimador deve ser fornecido para a votação ponderada.")

        estimadores_sklearn = [
            (nome, estimador) for nome, estimador in estimadores_elegiveis
        ]
        pesos = [
            float(self._pesos_por_modelo.get(nome, 1.0 / len(estimadores_elegiveis)))
            for nome, _ in estimadores_elegiveis
        ]

        return VotingRegressor(  # type: ignore[return-value]
            estimators=estimadores_sklearn,
            weights=pesos,
            n_jobs=-1,
        )
