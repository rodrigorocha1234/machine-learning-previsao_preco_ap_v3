from collections.abc import Callable
from typing import Final

import pandas as pd


class ViolacaoContratoDadosErro(Exception):
    def __init__(self, mensagem: str) -> None:
        super().__init__(mensagem)


RegraDados = Callable[[pd.DataFrame], bool]


class ValidadorContrato:
    def __init__(self, target: str = "Valor_da_Venda") -> None:
        colunas_obrigatorias: tuple[str, ...] = (
            "Bairro",
            "Zona",
            "Quartos",
            "Banheiros",
            "Vagas_Garagem",
            "Metragem",
            target,
        )
        self._colunas_obrigatorias: Final[tuple[str, ...]] = colunas_obrigatorias
        self._target: Final[str] = target

        self._regras: Final[tuple[tuple[RegraDados, str], ...]] = (
            (lambda df: not df.empty, "DataFrame nao pode estar vazio"),
            (
                lambda df: all(col in df.columns for col in self._colunas_obrigatorias),
                f"Todas as colunas obrigatorias devem estar presentes: {self._colunas_obrigatorias}",
            ),
            (
                lambda df: bool((df[self._target] > 0).all()),
                f"Valores da coluna target '{self._target}' devem ser estritamente positivos",
            ),
            (
                lambda df: bool((df["Metragem"] > 0).all()),
                "Valores de Metragem devem ser estritamente positivos",
            ),
            (
                lambda df: (
                    int(df[list(self._colunas_obrigatorias)].isnull().sum().sum()) == 0
                ),
                "Colunas obrigatorias nao devem conter valores nulos",
            ),
        )

    def validar(self, dados: pd.DataFrame) -> None:
        def verificar(regra_item: tuple[RegraDados, str]) -> None:
            regra, mensagem = regra_item
            assert regra(dados), mensagem

        try:
            tuple(map(verificar, self._regras))
        except AssertionError as erro:
            raise ViolacaoContratoDadosErro(
                f"Falha de contrato nos dados: {erro}"
            ) from erro
