from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias, runtime_checkable

CaminhoArquivo: TypeAlias = Path | str


@dataclass(frozen=True)
class ConfiguracaoDados:
    target: str
    grupos: tuple[str, ...]
    proporcao_holdout: float
    semente_holdout: int


@dataclass(frozen=True)
class ConfiguracaoValidacao:
    splits_externos: int
    repeticoes_externas: int
    semente_externa: int
    splits_internos: int
    embaralhar_interno: bool
    semente_interna: int


@dataclass(frozen=True)
class ConfiguracaoAvaliacao:
    metrica_principal: str
    metricas: tuple[str, ...]


@dataclass(frozen=True)
class ConfiguracaoSelecao:
    usar_votacao: bool
    quantidade_modelos: int
    criterio_unico: str


@dataclass(frozen=True)
class ConfiguracaoAmostra:
    minima_zona: int
    minima_bairro: int


@dataclass(frozen=True)
class ConfiguracaoMlflow:
    registrar_tuning: bool
    registrar_parametros: bool
    registrar_interpretacao_negocio: bool
    registrar_artefatos: bool


@dataclass(frozen=True)
class ConfiguracaoGeral:
    dados: ConfiguracaoDados
    validacao: ConfiguracaoValidacao
    avaliacao: ConfiguracaoAvaliacao
    selecao: ConfiguracaoSelecao
    amostra: ConfiguracaoAmostra
    mlflow: ConfiguracaoMlflow


@runtime_checkable
class ContratoConfiguracao(Protocol):
    def carregar_geral(self, caminho_pipeline: CaminhoArquivo) -> ConfiguracaoGeral: ...

    def carregar_modelos(
        self, caminho_modelos: CaminhoArquivo
    ) -> dict[str, dict[str, object]]: ...
