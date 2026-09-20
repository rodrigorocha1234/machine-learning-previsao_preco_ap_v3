"""Fábrica para instanciação de carregadores de dados."""

from ..enums_pkg.tipo_carregador import TipoCarregador
from .carregador_base import CarregadorBase
from .carregador_csv import CarregadorCsv
from .carregador_excel import CarregadorExcel
from .carregador_banco import CarregadorBanco


class FabricaCarregadores:
    """Factory para criação de instâncias de CarregadorBase a partir do TipoCarregador."""

    def criar(self, tipo: TipoCarregador) -> CarregadorBase:
        """Cria o carregador correspondente ao tipo informado.

        Parameters
        ----------
        tipo : TipoCarregador
            Tipo de carregador desejado.

        Returns
        -------
        CarregadorBase
            Instância concreta do carregador.
        """
        match tipo:
            case TipoCarregador.CSV:
                return CarregadorCsv()
            case TipoCarregador.EXCEL:
                return CarregadorExcel()
            case TipoCarregador.BANCO:
                return CarregadorBanco()
