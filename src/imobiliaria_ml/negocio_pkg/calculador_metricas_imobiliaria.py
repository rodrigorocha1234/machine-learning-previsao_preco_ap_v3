"""Calculador de métricas imobiliárias no conjunto holdout."""

import numpy as np
import pandas as pd
from .calculador_desconto_seguro import CalculadorDescontoSeguro
from .metricas_negocio_resultado import MetricasNegocioResultado


class CalculadorMetricasImobiliaria:
    """Avalia o desempenho financeiro e de risco da precificação imobiliária."""

    def __init__(self) -> None:
        self._calculador_desconto = CalculadorDescontoSeguro()

    def calcular_metricas(
        self,
        X_holdout: pd.DataFrame,
        y_real: pd.Series,
        y_pred: np.ndarray,
    ) -> MetricasNegocioResultado:
        """Calcula cobertura de erro, viés, tolerâncias, erros por faixa e por zona.

        Parameters
        ----------
        X_holdout : pd.DataFrame
            Conjunto de características do holdout (contendo 'Zona' quando disponível).
        y_real : pd.Series
            Valores reais observados de venda.
        y_pred : np.ndarray
            Valores previstos pelo modelo campeão.

        Returns
        -------
        MetricasNegocioResultado
            Resultado consolidado das métricas de negócio.
        """
        y_true_arr = np.asarray(y_real, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)

        erros_absolutos = np.abs(y_pred_arr - y_true_arr)
        erros_relativos = erros_absolutos / np.maximum(y_true_arr, 1.0)

        cobertura_5 = float(np.mean(erros_relativos <= 0.05))
        cobertura_10 = float(np.mean(erros_relativos <= 0.10))
        cobertura_15 = float(np.mean(erros_relativos <= 0.15))

        mae_reais = float(np.mean(erros_absolutos))
        vies_medio = float(np.mean(y_pred_arr - y_true_arr))

        # Riscos de subavaliação e sobreavaliação extrema (> 15%)
        risco_subprecificacao = float(np.mean((y_pred_arr - y_true_arr) / y_true_arr < -0.15))
        risco_superprecificacao = float(np.mean((y_pred_arr - y_true_arr) / y_true_arr > 0.15))

        desconto_seguro = self._calculador_desconto.calcular_desconto_seguro(
            mae_validado=mae_reais,
            preco_medio=float(np.mean(y_true_arr)),
        )

        df_analise = pd.DataFrame(
            {
                "real": y_true_arr,
                "previsto": y_pred_arr,
                "erro_absoluto": erros_absolutos,
                "erro_relativo": erros_relativos,
            }
        )

        # Erro por faixa de preço
        faixas = [0, 300_000, 600_000, 1_000_000, float("inf")]
        rotulos_faixas = [
            "Até R$ 300k",
            "R$ 300k - 600k",
            "R$ 600k - 1M",
            "Acima de R$ 1M",
        ]
        df_analise["faixa_preco"] = pd.cut(
            df_analise["real"], bins=faixas, labels=rotulos_faixas, right=True
        )

        tabela_faixas = (
            df_analise.groupby("faixa_preco", observed=False)
            .agg(
                quantidade=("real", "count"),
                mae=("erro_absoluto", "mean"),
                erro_relativo_medio=("erro_relativo", "mean"),
            )
            .reset_index()
        )

        # Erro por Zona se presente
        if "Zona" in X_holdout.columns:
            df_analise["Zona"] = X_holdout["Zona"].values
            tabela_zona = (
                df_analise.groupby("Zona", observed=False)
                .agg(
                    quantidade=("real", "count"),
                    mae=("erro_absoluto", "mean"),
                    erro_relativo_medio=("erro_relativo", "mean"),
                    vies=("previsto", lambda p: float(np.mean(p - df_analise.loc[p.index, "real"]))),
                )
                .reset_index()
            )
        else:
            tabela_zona = pd.DataFrame()

        return MetricasNegocioResultado(
            cobertura_5=cobertura_5,
            cobertura_10=cobertura_10,
            cobertura_15=cobertura_15,
            mae_reais=mae_reais,
            vies_medio=vies_medio,
            desconto_seguro_recomendado=desconto_seguro,
            risco_subprecificacao=risco_subprecificacao,
            risco_superprecificacao=risco_superprecificacao,
            tabela_erro_por_faixa=tabela_faixas,
            tabela_erro_por_zona=tabela_zona,
        )
