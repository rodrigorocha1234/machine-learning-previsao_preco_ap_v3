"""Validador com RepeatedKFold obrigatório de 30 repetições."""

import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from .calculador_metricas import CalculadorMetricas
from .resultado_validacao_cruzada import ResultadoValidacaoCruzada


class ValidadorRepeatedKFold:
    """Executa RepeatedKFold garantindo as mesmas partições para todos os modelos."""

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 30,
        random_state: int = 42,
    ) -> None:
        self._n_splits = n_splits
        self._n_repeats = n_repeats
        self._random_state = random_state
        self._calculador = CalculadorMetricas()

    def validar_modelos(
        self,
        modelos_pipelines: dict[str, Pipeline],
        X_treino: pd.DataFrame,
        y_treino: pd.Series,
    ) -> ResultadoValidacaoCruzada:
        """Avalia múltiplos pipelines nos mesmos folds e agrega os resultados por repetição.

        Parameters
        ----------
        modelos_pipelines : dict[str, Pipeline]
            Dicionário de nome do modelo -> Pipeline ajustada com melhores hiperparâmetros.
        X_treino : pd.DataFrame
            Conjunto de características.
        y_treino : pd.Series
            Conjunto de valores alvo.

        Returns
        -------
        ResultadoValidacaoCruzada
            Estrutura contendo tabela com todos os folds e matriz 30xN para testes estatísticos.
        """
        rkf = RepeatedKFold(
            n_splits=self._n_splits,
            n_repeats=self._n_repeats,
            random_state=self._random_state,
        )

        # Pré-gera os índices de partição para assegurar partição idêntica
        divisoes = list(rkf.split(X_treino, y_treino))

        registros_folds: list[dict[str, float | int | str]] = []

        for nome_modelo, pipeline in modelos_pipelines.items():
            for idx_split, (idx_train, idx_val) in enumerate(divisoes):
                repeticao = idx_split // self._n_splits
                fold = idx_split % self._n_splits

                X_tr, X_va = X_treino.iloc[idx_train], X_treino.iloc[idx_val]
                y_tr, y_va = y_treino.iloc[idx_train], y_treino.iloc[idx_val]

                estimador = clone(pipeline)

                t_ini_fit = time.time()
                estimador.fit(X_tr, y_tr)
                tempo_fit = float(time.time() - t_ini_fit)

                t_ini_pred = time.time()
                y_pred = estimador.predict(X_va)
                tempo_pred = float(time.time() - t_ini_pred)

                metricas = self._calculador.calcular(y_va, y_pred)

                registros_folds.append(
                    {
                        "modelo": nome_modelo,
                        "repeticao": repeticao,
                        "fold": fold,
                        "rmse": metricas.rmse,
                        "mae": metricas.mae,
                        "mse": metricas.mse,
                        "r2": metricas.r2,
                        "medae": metricas.medae,
                        "tempo_treino": tempo_fit,
                        "tempo_predicao": tempo_pred,
                    }
                )

        df_folds = pd.DataFrame(registros_folds)

        # Agregação por repetição (média dos folds para cada repetição)
        # Matriz: linhas = 30 repetições, colunas = modelos, valor = RMSE médio da repetição
        df_agregado = (
            df_folds.groupby(["repeticao", "modelo"])["rmse"]
            .mean()
            .unstack(level="modelo")
            .reset_index(drop=True)
        )

        # Resumo consolidado de métricas por modelo
        resumo = (
            df_folds.groupby("modelo")
            .agg(
                rmse_medio=("rmse", "mean"),
                rmse_desvio=("rmse", "std"),
                mae_medio=("mae", "mean"),
                mae_desvio=("mae", "std"),
                r2_medio=("r2", "mean"),
                r2_desvio=("r2", "std"),
                tempo_treino_medio=("tempo_treino", "mean"),
            )
            .sort_values(by="rmse_medio", ascending=True)
            .reset_index()
        )

        return ResultadoValidacaoCruzada(
            tabela_folds=df_folds,
            matriz_repeticoes=df_agregado,
            resumo_modelos=resumo,
        )
