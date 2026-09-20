"""Adaptador genérico para conversão entre modelos locais e distribuídos."""

from typing import Generic, TypeVar
from ..modelos_pkg.regressor_protocol import RegressorProtocol

TModeloOrigem = TypeVar("TModeloOrigem", bound=RegressorProtocol)
TModeloDestino = TypeVar("TModeloDestino", bound=RegressorProtocol)


class AdaptadorDistribuido(Generic[TModeloOrigem, TModeloDestino]):
    """Padrão GoF Adapter tipado para conversão entre ecossistemas de modelagem."""

    def adaptar(self, modelo: TModeloOrigem) -> TModeloDestino:
        """Adapta o estimador de origem para o contrato do estimador de destino.

        Parameters
        ----------
        modelo : TModeloOrigem
            Estimador scikit-learn local.

        Returns
        -------
        TModeloDestino
            Estimador adaptado (ex: wrapper distribuído Spark MLlib / Ray).
        """
        # Em modo local/compatibilidade direta, preserva a interface de regressão
        return modelo  # type: ignore[return-value]
