"""Estratégia de VotingRegressor com média simples dos estimadores."""

from sklearn.ensemble import VotingRegressor
from ..modelos_pkg.regressor_protocol import RegressorProtocol
from .estrategia_ensemble import EstrategiaEnsemble


class VotacaoRegressor(EstrategiaEnsemble):
    """Ensemble por votação simples (média aritmética)."""

    def criar_estimador(
        self,
        estimadores_elegiveis: list[tuple[str, RegressorProtocol]],
    ) -> RegressorProtocol:
        """Cria um VotingRegressor do scikit-learn sem pesos."""
        if not estimadores_elegiveis:
            raise ValueError("Ao menos um estimador deve ser fornecido para o VotingRegressor.")

        # Cast estrutural compativel com sklearn VotingRegressor
        estimadores_sklearn = [
            (nome, estimador) for nome, estimador in estimadores_elegiveis
        ]
        return VotingRegressor(estimators=estimadores_sklearn, n_jobs=-1)  # type: ignore[return-value]
