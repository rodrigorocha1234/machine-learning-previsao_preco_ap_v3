from collections.abc import Callable, Mapping
from typing import Final

from app_build.ajuste_modelos.contrato_tuning import ContratoTuning
from app_build.ajuste_modelos.estrategia_aleatoria import EstrategiaAleatoria
from app_build.ajuste_modelos.estrategia_grade import EstrategiaGrade
from app_build.ajuste_modelos.estrategia_nula import EstrategiaNula

CriadorEstrategia = Callable[[], ContratoTuning]

REGISTRO_TUNING: Final[Mapping[str, CriadorEstrategia]] = {
    "grid": EstrategiaGrade,
    "random": EstrategiaAleatoria,
    "nenhum": EstrategiaNula,
}


class FabricaTuning:
    @staticmethod
    def obter_estrategia(nome_estrategia: str) -> ContratoTuning:
        try:
            criador = REGISTRO_TUNING[nome_estrategia]
            return criador()
        except KeyError as erro:
            raise ValueError(
                f"Estrategia de tuning nao reconhecida: '{nome_estrategia}'"
            ) from erro
