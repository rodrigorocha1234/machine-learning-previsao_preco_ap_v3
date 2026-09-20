"""Gerador de relatórios interpretativos de negócio para modelos imobiliários."""

import pandas as pd


class InterpretadorModelo:
    """Traduz coeficientes técnicos em interpretações econômicas para o negócio."""

    def gerar_interpretacao_markdown(
        self,
        tabela_coeficientes: pd.DataFrame,
        intercepto: float,
    ) -> str:
        """Produz texto analítico em formato Markdown com a interpretação de cada variável.

        Parameters
        ----------
        tabela_coeficientes : pd.DataFrame
            DataFrame com as colunas feature, coeficiente, sinal, magnitude.
        intercepto : float
            Valor base estimado pelo modelo.

        Returns
        -------
        str
            Documento Markdown estruturado.
        """
        linhas: list[str] = [
            "# Interpretação de Negócio do Modelo Linear\n",
            f"**Valor-base (Intercepto):** R$ {intercepto:,.2f}\n",
            "## Efeitos Marginais dos Atributos\n",
        ]

        for _, row in tabela_coeficientes.iterrows():
            nome = str(row["feature"])
            coef = float(row["coeficiente"])
            magnitude = float(row["magnitude"])

            if "cat__Zona" in nome or "Zona_" in nome:
                linhas.append(
                    f"- **{nome}**: Ajuste regional de **R$ {coef:+,.2f}** no preço final "
                    f"em comparação com a categoria base de referência, "
                    f"mantidas fixas as demais características físicas do imóvel.\n"
                )
            else:
                sentido = "acréscimo" if coef >= 0 else "decréscimo"
                linhas.append(
                    f"- **{nome}**: Mantidas todas as demais variáveis constantes, cada unidade adicional "
                    f"está associada a um **{sentido} médio de R$ {magnitude:,.2f}** "
                    f"no valor previsto do imóvel.\n"
                )

        return "\n".join(linhas)
