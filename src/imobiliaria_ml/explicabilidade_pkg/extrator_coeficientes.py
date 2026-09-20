"""Extrator de coeficientes e equações matemáticas para modelos lineares."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class ExtratorCoeficientes:
    """Extrai coeficientes, intercepto e equações analíticas de pipelines lineares."""

    def extrair(
        self,
        pipeline_ajustada: Pipeline,
    ) -> tuple[pd.DataFrame, float, str]:
        """Extrai a tabela de coeficientes, o intercepto e a equação formatada.

        Parameters
        ----------
        pipeline_ajustada : Pipeline
            Pipeline treinada contendo pré-processador e estimador linear.

        Returns
        -------
        tuple[pd.DataFrame, float, str]
            Tabela de coeficientes, intercepto numérico e equação em texto puro.
        """
        preprocessador = pipeline_ajustada.named_steps.get("pre") or pipeline_ajustada.steps[0][1]
        estimador = pipeline_ajustada.named_steps.get("modelo") or pipeline_ajustada.steps[-1][1]

        # Se for pipeline aninhada (ex: regressão polinomial)
        if isinstance(estimador, Pipeline):
            estimador_final = estimador.named_steps.get("linear") or estimador.steps[-1][1]
        else:
            estimador_final = estimador

        if not hasattr(estimador_final, "coef_"):
            raise ValueError("O estimador fornecido não possui o atributo 'coef_'.")

        coeficientes: np.ndarray = np.ravel(estimador_final.coef_)
        intercepto: float = float(getattr(estimador_final, "intercept_", 0.0))

        # Extrai nomes das features transformadas
        nomes_features: list[str] = []
        if hasattr(preprocessador, "get_feature_names_out"):
            nomes_features = [str(n) for n in preprocessador.get_feature_names_out()]
        else:
            nomes_features = [f"x{i}" for i in range(len(coeficientes))]

        # Ajuste de tamanho caso regressão polinomial tenha expandido as features
        if len(nomes_features) != len(coeficientes):
            if hasattr(estimador, "named_steps") and "poly" in estimador.named_steps:
                poly = estimador.named_steps["poly"]
                if hasattr(poly, "get_feature_names_out"):
                    nomes_features = [str(n) for n in poly.get_feature_names_out(nomes_features)]
                else:
                    nomes_features = [f"poly_{i}" for i in range(len(coeficientes))]
            else:
                nomes_features = [f"feature_{i}" for i in range(len(coeficientes))]

        registros = []
        termos_equacao = [f"{intercepto:,.2f}"]

        for nome, coef in zip(nomes_features, coeficientes):
            coef_val = float(coef)
            sinal = "+" if coef_val >= 0 else "-"
            magnitude = float(abs(coef_val))
            registros.append(
                {
                    "feature": nome,
                    "coeficiente": coef_val,
                    "sinal": sinal,
                    "magnitude": magnitude,
                }
            )
            termos_equacao.append(f"({sinal} {magnitude:,.2f} * {nome})")

        df_coef = (
            pd.DataFrame(registros)
            .sort_values(by="magnitude", ascending=False)
            .reset_index(drop=True)
        )

        equacao = "Valor_da_Venda = " + " + ".join(termos_equacao)
        return df_coef, intercepto, equacao
