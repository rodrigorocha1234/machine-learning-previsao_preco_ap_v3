"""Validador de qualidade, completude e duplicatas nos dados."""

import pandas as pd
from .resultado_validacao import ResultadoValidacao


class ValidadorQualidade:
    """Validador de qualidade de registros imobiliários."""

    def __init__(self, limite_taxa_nulos: float = 0.30) -> None:
        """Inicializa o validador.

        Parameters
        ----------
        limite_taxa_nulos : float
            Taxa máxima tolerada de nulos por coluna antes de emitir erro (padrão: 30%).
        """
        self._limite_taxa_nulos = limite_taxa_nulos

    def validar(self, dados: pd.DataFrame) -> ResultadoValidacao:
        """Valida a qualidade dos dados de imóveis.

        Parameters
        ----------
        dados : pd.DataFrame
            DataFrame de entrada.

        Returns
        -------
        ResultadoValidacao
            Resultado estruturado da validação de qualidade.
        """
        erros: list[str] = []
        avisos: list[str] = []
        metricas: dict[str, float | int | str] = {}

        total_linhas = len(dados)
        if total_linhas == 0:
            return ResultadoValidacao(
                valido=False,
                mensagens_erro=("DataFrame vazio fornecido.",),
            )

        # Checagem de duplicatas
        duplicatas = int(dados.duplicated().sum())
        metricas["duplicatas_totais"] = duplicatas
        if duplicatas > 0:
            avisos.append(
                f"Detectadas {duplicatas} linhas duplicadas perfeitas ({duplicatas / total_linhas:.2%})."
            )

        # Checagem de valores ausentes por coluna
        for coluna in dados.columns:
            nulos = int(dados[coluna].isna().sum())
            taxa_nulos = nulos / total_linhas
            metricas[f"nulos_{coluna}"] = nulos
            metricas[f"taxa_nulos_{coluna}"] = taxa_nulos

            if taxa_nulos > self._limite_taxa_nulos:
                erros.append(
                    f"Coluna {coluna} ultrapassou o limite de nulos ({taxa_nulos:.1%} > {self._limite_taxa_nulos:.1%})."
                )
            elif nulos > 0:
                avisos.append(
                    f"Coluna {coluna} possui {nulos} valores ausentes ({taxa_nulos:.1%})."
                )

        valido = len(erros) == 0
        return ResultadoValidacao(
            valido=valido,
            mensagens_erro=tuple(erros),
            mensagens_aviso=tuple(avisos),
            metricas=metricas,
        )
