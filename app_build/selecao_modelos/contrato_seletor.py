from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app_build.estatistica_modelos.contrato_estatistica import ResultadoFriedman
from app_build.validacao_cruzada.contrato_validador import ResultadoNestedCv


@dataclass(frozen=True)
class DecisaoSelecao:
    modelo_principal: str
    modelos_selecionados: tuple[str, ...]
    modo_selecao: str
    justificativa: str

    @property
    def nome_modelo_final(self) -> str:
        return {
            "MODELO_UNICO": self.modelo_principal,
            "ENSEMBLE_VOTACAO": "voting_regressor",
        }[self.modo_selecao]


@runtime_checkable
class ContratoSeletor(Protocol):
    def selecionar(
        self,
        resultados_cv: Mapping[str, ResultadoNestedCv],
        resultado_friedman: ResultadoFriedman,
    ) -> DecisaoSelecao: ...
