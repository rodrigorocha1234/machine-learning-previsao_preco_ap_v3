"""Validador de conformidade do esquema de dados."""

import pandas as pd
from .resultado_validacao import ResultadoValidacao


class ValidadorEsquema:
    """Validador das colunas obrigatórias, tipos e limites numéricos."""

    COLUNAS_CARACTERISTICAS: tuple[str, ...] = (
        "Zona",
        "Quartos",
        "Banheiros",
        "Vagas",
        "Metragem",
    )
    COLUNA_ALVO: str = "Valor_da_Venda"

    def validar(
        self,
        dados: pd.DataFrame,
        exigir_alvo: bool = True,
    ) -> ResultadoValidacao:
        """Valida se o DataFrame cumpre os requisitos de esquema.

        Parameters
        ----------
        dados : pd.DataFrame
            DataFrame de imóveis a validar.
        exigir_alvo : bool
            Indica se a coluna Valor_da_Venda é obrigatória (treino) ou opcional (inferência).

        Returns
        -------
        ResultadoValidacao
            Estrutura contendo o status e possíveis inconsistências.
        """
        erros: list[str] = []
        avisos: list[str] = []

        # Validação de presença de colunas
        colunas_necessarias = list(self.COLUNAS_CARACTERISTICAS)
        if exigir_alvo:
            colunas_necessarias.append(self.COLUNA_ALVO)

        for col in colunas_necessarias:
            if col not in dados.columns:
                erros.append(f"Coluna obrigatória ausente: {col}")

        if erros:
            return ResultadoValidacao(
                valido=False,
                mensagens_erro=tuple(erros),
                mensagens_aviso=tuple(avisos),
            )

        # Validação de limites numéricos
        for col_inteira in ("Quartos", "Banheiros", "Vagas"):
            valores = dados[col_inteira].dropna()
            if (valores < 0).any():
                erros.append(f"Coluna {col_inteira} contém valores negativos.")

        # Metragem deve ser estritamente maior que 0
        metragens = dados["Metragem"].dropna()
        if (metragens <= 0).any():
            erros.append("Coluna Metragem contém valores menores ou iguais a zero.")

        # Valor_da_Venda deve ser estritamente maior que 0 quando presente
        if self.COLUNA_ALVO in dados.columns:
            valores_venda = dados[self.COLUNA_ALVO].dropna()
            if (valores_venda <= 0).any():
                erros.append("Coluna Valor_da_Venda contém valores menores ou iguais a zero.")

        valido = len(erros) == 0
        return ResultadoValidacao(
            valido=valido,
            mensagens_erro=tuple(erros),
            mensagens_aviso=tuple(avisos),
            metricas={
                "total_linhas": len(dados),
                "total_colunas": len(dados.columns),
            },
        )
