import time
from collections.abc import Mapping
from typing import Final, cast, override

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app_build.ajuste_modelos.fabrica_estimadores import FabricaEstimadores
from app_build.ajuste_modelos.fabrica_tuning import FabricaTuning
from app_build.processamento_dados.construtor_pipeline import ConstrutorPipeline
from app_build.validacao_cruzada.acumulador_metricas import AcumuladorMetricas
from app_build.validacao_cruzada.contrato_validador import (
    ContratoValidador,
    ResultadoFoldExterno,
    ResultadoNestedCv,
)
from app_build.validacao_cruzada.particionador_interno import ParticionadorInterno


class AvaliadorAninhado(ContratoValidador):
    def __init__(
        self,
        configuracao_modelo: dict[str, object],
        divisoes_externas: tuple[tuple[np.ndarray, np.ndarray], ...],
        particionador_interno: ParticionadorInterno | None = None,
        construtor_pipeline: ConstrutorPipeline | None = None,
    ) -> None:
        self._config: Final[dict[str, object]] = configuracao_modelo
        self._divisoes_externas: Final[tuple[tuple[np.ndarray, np.ndarray], ...]] = (
            divisoes_externas
        )
        self._particionador_interno: Final[ParticionadorInterno] = (
            particionador_interno or ParticionadorInterno()
        )
        self._construtor_pipeline: Final[ConstrutorPipeline] = (
            construtor_pipeline or ConstrutorPipeline()
        )

    @override
    def avaliar_modelo(
        self,
        nome_modelo: str,
        dados_x: pd.DataFrame,
        vetor_y: pd.Series,
    ) -> ResultadoNestedCv:
        parametros_base = dict(
            cast(Mapping[str, object], self._config.get("parametros", {}))
        )
        secao_tuning = dict(cast(Mapping[str, object], self._config.get("tuning", {})))
        nome_estrategia = str(secao_tuning.get("estrategia", "nenhum"))
        espaco_params = dict(
            cast(Mapping[str, object], secao_tuning.get("parametros", {}))
        )
        metrica_scoring = str(
            secao_tuning.get("scoring", "neg_root_mean_squared_error")
        )
        n_iter = int(cast(int, secao_tuning.get("n_iter", 10)))
        semente = int(cast(int, secao_tuning.get("random_state", 42)))

        estrategia_tuning = FabricaTuning.obter_estrategia(nome_estrategia)
        validador_interno = self._particionador_interno.obter_validador()

        def processar_fold(
            item: tuple[int, tuple[np.ndarray, np.ndarray]],
        ) -> ResultadoFoldExterno:
            idx_fold, (idx_treino, idx_teste) = item
            inicio = time.perf_counter()

            x_treino = dados_x.iloc[idx_treino].copy()
            y_treino = vetor_y.iloc[idx_treino].copy()
            x_teste = dados_x.iloc[idx_teste].copy()
            y_teste = vetor_y.iloc[idx_teste].copy()

            estimador_base = FabricaEstimadores.criar_estimador(
                nome_modelo, parametros_base
            )
            preprocessador = self._construtor_pipeline.criar_preprocessador()
            pipeline_base = Pipeline(
                [
                    ("preprocessamento", preprocessador),
                    ("modelo", estimador_base),
                ]
            )

            resultado_tuning = estrategia_tuning.executar_tuning(
                estimador_base=pipeline_base,
                espaco_parametros=espaco_params,
                dados_x=x_treino,
                vetor_y=y_treino,
                validador_cv=validador_interno,
                metrica_scoring=metrica_scoring,
                semente=semente,
                iteracoes=n_iter,
            )

            previsoes = resultado_tuning.melhor_estimador.predict(x_teste)
            valores_reais = y_teste.to_numpy()
            residuos = valores_reais - previsoes
            metricas = AcumuladorMetricas.calcular_metricas(valores_reais, previsoes)
            duracao = time.perf_counter() - inicio

            return ResultadoFoldExterno(
                indice_fold=idx_fold,
                metricas=metricas,
                melhores_parametros=resultado_tuning.melhores_parametros,
                previsoes=previsoes,
                valores_reais=valores_reais,
                residuos=residuos,
                tempo_segundos=duracao,
            )

        folds_indexados = tuple(enumerate(self._divisoes_externas))
        resultados_folds = tuple(map(processar_fold, folds_indexados))

        metricas_por_fold = tuple(r.metricas for r in resultados_folds)
        metricas_medias = AcumuladorMetricas.agregar_medias(metricas_por_fold)
        metricas_medianas = AcumuladorMetricas.agregar_medianas(metricas_por_fold)
        residuos_totais = np.concatenate([r.residuos for r in resultados_folds])

        return ResultadoNestedCv(
            nome_modelo=nome_modelo,
            resultados_folds=resultados_folds,
            metricas_medias=metricas_medias,
            metricas_medianas=metricas_medianas,
            residuos_totais=residuos_totais,
        )
