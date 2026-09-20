"""Fábrica para instanciar estratégias de escalonamento numérico."""

from ..enums_pkg.tipo_escalonador import TipoEscalonador
from .estrategia_escalonamento import EstrategiaEscalonamento
from .escalonador_standard import EscalonadorStandard
from .escalonador_minmax import EscalonadorMinmax
from .escalonador_robusto import EscalonadorRobusto
from .escalonador_sem_escala import EscalonadorSemEscala


class FabricaEscalonadores:
    """Factory para criação de estratégias de escalonamento por TipoEscalonador."""

    def criar(self, tipo: TipoEscalonador) -> EstrategiaEscalonamento:
        """Retorna a estratégia de escalonamento correspondente.

        Parameters
        ----------
        tipo : TipoEscalonador
            Tipo de escalonamento desejado.

        Returns
        -------
        EstrategiaEscalonamento
            Instância concreta da estratégia de escalonamento.
        """
        match tipo:
            case TipoEscalonador.STANDARD:
                return EscalonadorStandard()
            case TipoEscalonador.MINMAX:
                return EscalonadorMinmax()
            case TipoEscalonador.ROBUSTO:
                return EscalonadorRobusto()
            case TipoEscalonador.SEM_ESCALA:
                return EscalonadorSemEscala()
