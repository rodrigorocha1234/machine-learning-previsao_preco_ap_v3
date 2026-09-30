from collections.abc import Callable
from typing import Final

from app_build.configuracao_sistema.contrato_configuracao import ConfiguracaoGeral


class ViolacaoEsquemaErro(Exception):
    def __init__(self, mensagem: str) -> None:
        super().__init__(mensagem)


RegraValidacao = Callable[[ConfiguracaoGeral], bool]


class ValidadorEsquema:
    def __init__(self) -> None:
        self._regras: Final[tuple[tuple[RegraValidacao, str], ...]] = (
            (
                lambda cfg: len(cfg.dados.target.strip()) > 0,
                "Target não pode ser vazio",
            ),
            (
                lambda cfg: 0.0 < cfg.dados.proporcao_holdout < 1.0,
                "Proporção de holdout deve estar entre 0 e 1",
            ),
            (
                lambda cfg: cfg.validacao.splits_externos >= 2,
                "Splits externos devem ser >= 2",
            ),
            (
                lambda cfg: cfg.validacao.splits_internos >= 2,
                "Splits internos devem ser >= 2",
            ),
            (
                lambda cfg: cfg.validacao.repeticoes_externas >= 1,
                "Repetições externas devem ser >= 1",
            ),
            (
                lambda cfg: cfg.avaliacao.metrica_principal in cfg.avaliacao.metricas,
                "Métrica principal deve constar na lista de métricas",
            ),
            (
                lambda cfg: cfg.amostra.minima_zona >= 1,
                "Mínimo de amostra por zona deve ser >= 1",
            ),
            (
                lambda cfg: cfg.amostra.minima_bairro >= 1,
                "Mínimo de amostra por bairro deve ser >= 1",
            ),
        )

    def validar(self, configuracao: ConfiguracaoGeral) -> None:
        def verificar_regra(item: tuple[RegraValidacao, str]) -> None:
            regra, mensagem = item
            assert regra(configuracao), mensagem

        try:
            tuple(map(verificar_regra, self._regras))
        except AssertionError as falha:
            raise ViolacaoEsquemaErro(
                f"Esquema de configuracao invalido: {falha}"
            ) from falha
