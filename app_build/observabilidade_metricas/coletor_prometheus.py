import time
from typing import Final, override

from prometheus_client import Counter, Gauge, Histogram, Info

from app_build.observabilidade_metricas.contrato_telemetria import ContratoTelemetria


class ColetorPrometheus(ContratoTelemetria):
    def __init__(self) -> None:
        # ── Inferência ──────────────────────────────────────────────────────────
        self._contador_predicoes: Final[Counter] = Counter(
            "apartamentos_predicoes_total",
            "Total de inferencias de precos imobiliarios realizadas",
            ["zona"],
        )
        self._histograma_latencia: Final[Histogram] = Histogram(
            "apartamentos_latencia_segundos",
            "Latencia da estimativa em segundos",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
        )
        self._gauge_valor_medio: Final[Gauge] = Gauge(
            "apartamentos_valor_medio_previsto",
            "Media recente dos precos previstos",
            ["zona"],
        )

        # ── Métricas do Modelo ──────────────────────────────────────────────────
        self._gauge_rmse: Final[Gauge] = Gauge(
            "apartamentos_modelo_rmse",
            "RMSE do modelo campeao em producao",
        )
        self._gauge_mae: Final[Gauge] = Gauge(
            "apartamentos_modelo_mae",
            "MAE do modelo campeao em producao",
        )
        self._gauge_r2: Final[Gauge] = Gauge(
            "apartamentos_modelo_r2",
            "R2 do modelo campeao em producao",
        )
        self._gauge_mape: Final[Gauge] = Gauge(
            "apartamentos_modelo_mape",
            "MAPE percentual absoluto medio do modelo campeao",
        )
        self._gauge_amostras_treino: Final[Gauge] = Gauge(
            "apartamentos_modelo_total_amostras_treino",
            "Total de amostras utilizadas no treino do modelo campeao",
        )
        self._gauge_amostras_holdout: Final[Gauge] = Gauge(
            "apartamentos_modelo_total_amostras_holdout",
            "Total de amostras do conjunto holdout de avaliacao",
        )
        self._info_modelo: Final[Info] = Info(
            "apartamentos_modelo_campeao",
            "Informacoes do modelo campeao registrado no MLflow",
        )

        # ── Drift ───────────────────────────────────────────────────────────────
        self._gauge_drift_psi_predicoes: Final[Gauge] = Gauge(
            "apartamentos_drift_psi_predicoes",
            "Indice PSI de drift das predicoes de preco",
        )
        self._gauge_drift_psi_area: Final[Gauge] = Gauge(
            "apartamentos_drift_psi_area",
            "Indice PSI de drift da Area Privativa m2",
        )
        self._gauge_drift_ks_stat: Final[Gauge] = Gauge(
            "apartamentos_drift_ks_area_stat",
            "Estatistica do teste Kolmogorov-Smirnov para Area Privativa",
        )
        self._gauge_drift_ks_pval: Final[Gauge] = Gauge(
            "apartamentos_drift_ks_area_pvalor",
            "P-valor do teste Kolmogorov-Smirnov para Area Privativa",
        )
        self._gauge_drift_wasserstein: Final[Gauge] = Gauge(
            "apartamentos_drift_wasserstein_area",
            "Distancia normalizada de Wasserstein para Area Privativa",
        )
        self._gauge_drift_status: Final[Gauge] = Gauge(
            "apartamentos_drift_status_geral",
            "Status geral de drift (0=Estavel, 1=Moderado, 2=Critico)",
        )
        self._gauge_drift_feature: Final[Gauge] = Gauge(
            "apartamentos_drift_psi_features",
            "Indice PSI de drift por feature de entrada",
            ["feature"],
        )
        self._gauge_desvio_preco_zona: Final[Gauge] = Gauge(
            "apartamentos_drift_desvio_preco_m2_zona",
            "Desvio percentual do preco por metro quadrado por zona vs baseline",
            ["zona"],
        )

        # ── Testes Estatísticos ─────────────────────────────────────────────────
        self._gauge_friedman_chi2: Final[Gauge] = Gauge(
            "apartamentos_estatistica_friedman_chi2",
            "Estatistica qui-quadrado de Friedman para comparacao de modelos",
        )
        self._gauge_friedman_pvalor: Final[Gauge] = Gauge(
            "apartamentos_estatistica_friedman_pvalor",
            "P-valor do teste de Friedman entre modelos concorrentes",
        )
        self._gauge_friedman_significativo: Final[Gauge] = Gauge(
            "apartamentos_estatistica_friedman_significativo",
            "Indicador se ha diferenca estatistica pelo teste de Friedman (1=Sim, 0=Nao)",
        )
        self._gauge_nemenyi_cd: Final[Gauge] = Gauge(
            "apartamentos_estatistica_nemenyi_cd",
            "Diferenca critica (CD) do teste post-hoc de Nemenyi",
        )
        self._gauge_rank_modelo: Final[Gauge] = Gauge(
            "apartamentos_estatistica_rank_medio_modelo",
            "Rank medio do modelo na validacao cruzada (menor rank = melhor)",
            ["modelo"],
        )
        self._gauge_nemenyi_dif_par: Final[Gauge] = Gauge(
            "apartamentos_estatistica_nemenyi_dif_ranks",
            "Diferenca absoluta de ranks entre par de modelos no teste Nemenyi",
            ["modelo_a", "modelo_b"],
        )
        self._gauge_nemenyi_par_significativo: Final[Gauge] = Gauge(
            "apartamentos_estatistica_nemenyi_par_significativo",
            "Diferenca estatistica significativa entre par no Nemenyi (1=Sim, 0=Nao)",
            ["modelo_a", "modelo_b"],
        )
        self._gauge_shapiro_pvalor: Final[Gauge] = Gauge(
            "apartamentos_estatistica_shapiro_pvalor",
            "P-valor do teste de normalidade Shapiro-Wilk para residuos do modelo",
            ["modelo"],
        )
        self._gauge_shapiro_normal: Final[Gauge] = Gauge(
            "apartamentos_estatistica_shapiro_eh_normal",
            "Indicador se os residuos seguem distribuicao normal (1=Sim, 0=Nao)",
            ["modelo"],
        )

        # ── Métricas de Pipeline ────────────────────────────────────────────────
        self._gauge_etapa_ativa: Final[Gauge] = Gauge(
            "apartamentos_pipeline_etapa_indice_atual",
            "Indice da etapa em execucao no momento (0 = idle)",
        )
        self._gauge_etapa_total: Final[Gauge] = Gauge(
            "apartamentos_pipeline_total_etapas",
            "Total de etapas configuradas na esteira",
        )
        self._contador_etapas_iniciadas: Final[Counter] = Counter(
            "apartamentos_pipeline_etapas_iniciadas_total",
            "Contador cumulativo de etapas iniciadas pelo ExecutorEsteira",
        )
        self._contador_etapas_concluidas: Final[Counter] = Counter(
            "apartamentos_pipeline_etapas_concluidas_total",
            "Contador cumulativo de etapas concluidas com sucesso",
        )
        self._contador_etapas_falhas: Final[Counter] = Counter(
            "apartamentos_pipeline_etapas_falhas_total",
            "Contador cumulativo de etapas encerradas com falha",
        )
        self._histograma_duracao_etapa: Final[Histogram] = Histogram(
            "apartamentos_pipeline_etapa_duracao_segundos",
            "Duracao de execucao de cada etapa do pipeline em segundos",
            ["etapa"],
            buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
        )
        self._gauge_pipeline_duracao_total: Final[Gauge] = Gauge(
            "apartamentos_pipeline_duracao_total_segundos",
            "Duracao total da ultima execucao completa do pipeline em segundos",
        )
        self._gauge_pipeline_ultimo_timestamp: Final[Gauge] = Gauge(
            "apartamentos_pipeline_ultima_execucao_timestamp",
            "Timestamp Unix da ultima execucao completa do pipeline",
        )
        self._contador_execucoes_pipeline: Final[Counter] = Counter(
            "apartamentos_pipeline_execucoes_total",
            "Total de execucoes completas do pipeline realizadas",
        )
        self._gauge_pipeline_etapas_com_falha: Final[Gauge] = Gauge(
            "apartamentos_pipeline_etapas_com_falha_ultima_execucao",
            "Quantidade de etapas com falha na ultima execucao do pipeline",
        )

        # ── Métricas de Negócio Hierárquico ────────────────────────────────────
        self._gauge_preco_mediano_global: Final[Gauge] = Gauge(
            "apartamentos_negocio_preco_mediano_global",
            "Preco mediano global dos imoveis no dataset (R$)",
        )
        self._gauge_preco_medio_global: Final[Gauge] = Gauge(
            "apartamentos_negocio_preco_medio_global",
            "Preco medio global dos imoveis no dataset (R$)",
        )
        self._gauge_preco_m2_mediano_global: Final[Gauge] = Gauge(
            "apartamentos_negocio_preco_m2_mediano_global",
            "Valor mediano do metro quadrado no dataset (R$/m2)",
        )
        self._gauge_total_imoveis: Final[Gauge] = Gauge(
            "apartamentos_negocio_total_imoveis_dataset",
            "Total de imoveis no dataset utilizado para treinamento",
        )
        self._gauge_preco_mediano_zona: Final[Gauge] = Gauge(
            "apartamentos_negocio_preco_mediano_zona",
            "Preco mediano dos imoveis por zona (R$)",
            ["zona"],
        )
        self._gauge_preco_m2_mediano_zona: Final[Gauge] = Gauge(
            "apartamentos_negocio_preco_m2_mediano_zona",
            "Valor mediano do metro quadrado por zona (R$/m2)",
            ["zona"],
        )
        self._gauge_amostras_zona: Final[Gauge] = Gauge(
            "apartamentos_negocio_total_amostras_zona",
            "Total de imoveis por zona no dataset",
            ["zona"],
        )
        self._gauge_preco_mediano_bairro: Final[Gauge] = Gauge(
            "apartamentos_negocio_preco_mediano_bairro",
            "Preco mediano dos imoveis por bairro (R$)",
            ["bairro", "zona"],
        )
        self._gauge_amostras_bairro: Final[Gauge] = Gauge(
            "apartamentos_negocio_total_amostras_bairro",
            "Total de imoveis por bairro no dataset",
            ["bairro", "zona"],
        )

    # ── Inferência ──────────────────────────────────────────────────────────────

    @override
    def registrar_inferencia(
        self, duracao_segundos: float, zona: str, valor_previsto: float
    ) -> None:
        zona_normalizada = str(zona).strip() or "Desconhecida"
        self._contador_predicoes.labels(zona=zona_normalizada).inc()
        self._histograma_latencia.observe(duracao_segundos)
        self._gauge_valor_medio.labels(zona=zona_normalizada).set(valor_previsto)

    # ── Modelo ──────────────────────────────────────────────────────────────────

    @override
    def atualizar_metricas_modelo(self, rmse: float, mae: float, r2: float) -> None:
        self._gauge_rmse.set(rmse)
        self._gauge_mae.set(mae)
        self._gauge_r2.set(r2)

    @override
    def atualizar_metricas_modelo_completo(
        self,
        rmse: float,
        mae: float,
        r2: float,
        mape: float,
        total_amostras_treino: int,
        total_amostras_holdout: int,
        versao_modelo: str,
        nome_modelo: str,
    ) -> None:
        self._gauge_rmse.set(rmse)
        self._gauge_mae.set(mae)
        self._gauge_r2.set(r2)
        self._gauge_mape.set(mape)
        self._gauge_amostras_treino.set(float(total_amostras_treino))
        self._gauge_amostras_holdout.set(float(total_amostras_holdout))
        self._info_modelo.info({"versao": str(versao_modelo), "nome": str(nome_modelo)})

    # ── Drift ───────────────────────────────────────────────────────────────────

    @override
    def registrar_drift_dados(
        self,
        psi_predicoes: float,
        psi_area: float,
        ks_stat_area: float,
        ks_pval_area: float,
        wasserstein_area: float,
        status_geral: int,
    ) -> None:
        self._gauge_drift_psi_predicoes.set(psi_predicoes)
        self._gauge_drift_psi_area.set(psi_area)
        self._gauge_drift_ks_stat.set(ks_stat_area)
        self._gauge_drift_ks_pval.set(ks_pval_area)
        self._gauge_drift_wasserstein.set(wasserstein_area)
        self._gauge_drift_status.set(float(status_geral))

    @override
    def registrar_drift_feature(self, feature: str, psi: float) -> None:
        nome_feature = str(feature).strip() or "Desconhecida"
        self._gauge_drift_feature.labels(feature=nome_feature).set(psi)

    @override
    def registrar_desvio_preco_zona(self, zona: str, desvio_percentual: float) -> None:
        zona_normalizada = str(zona).strip() or "Desconhecida"
        self._gauge_desvio_preco_zona.labels(zona=zona_normalizada).set(desvio_percentual)

    # ── Testes Estatísticos ─────────────────────────────────────────────────────

    @override
    def registrar_testes_estatisticos(
        self,
        friedman_chi2: float,
        friedman_pvalor: float,
        friedman_significativo: bool,
        nemenyi_cd: float,
        ranks_modelos: dict[str, float],
        comparacoes_nemenyi: list[tuple[str, str, float, bool]],
        resultados_shapiro: dict[str, tuple[float, bool]],
    ) -> None:
        self._gauge_friedman_chi2.set(friedman_chi2)
        self._gauge_friedman_pvalor.set(friedman_pvalor)
        self._gauge_friedman_significativo.set(float(friedman_significativo))
        self._gauge_nemenyi_cd.set(nemenyi_cd)
        for modelo, rank in ranks_modelos.items():
            self._gauge_rank_modelo.labels(modelo=modelo).set(rank)
        for mod_a, mod_b, dif, sig in comparacoes_nemenyi:
            self._gauge_nemenyi_dif_par.labels(modelo_a=mod_a, modelo_b=mod_b).set(dif)
            self._gauge_nemenyi_par_significativo.labels(
                modelo_a=mod_a, modelo_b=mod_b
            ).set(float(sig))
        for modelo, (pval, eh_norm) in resultados_shapiro.items():
            self._gauge_shapiro_pvalor.labels(modelo=modelo).set(pval)
            self._gauge_shapiro_normal.labels(modelo=modelo).set(float(eh_norm))

    # ── Pipeline ────────────────────────────────────────────────────────────────

    @override
    def registrar_inicio_etapa(self, nome_etapa: str, indice: int, total: int) -> None:
        self._gauge_etapa_ativa.set(indice)
        self._gauge_etapa_total.set(total)
        self._contador_etapas_iniciadas.inc()

    @override
    def registrar_fim_etapa(
        self,
        nome_etapa: str,
        indice: int,
        total: int,
        duracao_segundos: float,
        sucesso: bool,
    ) -> None:
        self._histograma_duracao_etapa.labels(etapa=nome_etapa).observe(duracao_segundos)
        if sucesso:
            self._contador_etapas_concluidas.inc()
        else:
            self._contador_etapas_falhas.inc()

    @override
    def registrar_pipeline_concluido(
        self,
        duracao_total_segundos: float,
        total_etapas: int,
        etapas_com_falha: int,
    ) -> None:
        self._gauge_pipeline_duracao_total.set(duracao_total_segundos)
        self._gauge_pipeline_ultimo_timestamp.set(time.time())
        self._gauge_etapa_ativa.set(0)
        self._contador_execucoes_pipeline.inc()
        self._gauge_pipeline_etapas_com_falha.set(float(etapas_com_falha))

    # ── Negócio Hierárquico ─────────────────────────────────────────────────────

    @override
    def registrar_metricas_negocio(
        self,
        preco_mediano_global: float,
        preco_medio_global: float,
        preco_m2_mediano_global: float,
        total_imoveis_dataset: int,
    ) -> None:
        self._gauge_preco_mediano_global.set(preco_mediano_global)
        self._gauge_preco_medio_global.set(preco_medio_global)
        self._gauge_preco_m2_mediano_global.set(preco_m2_mediano_global)
        self._gauge_total_imoveis.set(float(total_imoveis_dataset))

    @override
    def registrar_metricas_negocio_zona(
        self,
        zona: str,
        preco_mediano: float,
        preco_m2_mediano: float,
        total_amostras: int,
    ) -> None:
        zona_norm = str(zona).strip() or "Desconhecida"
        self._gauge_preco_mediano_zona.labels(zona=zona_norm).set(preco_mediano)
        self._gauge_preco_m2_mediano_zona.labels(zona=zona_norm).set(preco_m2_mediano)
        self._gauge_amostras_zona.labels(zona=zona_norm).set(float(total_amostras))

    @override
    def registrar_metricas_negocio_bairro(
        self,
        bairro: str,
        zona: str,
        preco_mediano: float,
        total_amostras: int,
    ) -> None:
        bairro_norm = str(bairro).strip() or "Desconhecido"
        zona_norm = str(zona).strip() or "Desconhecida"
        self._gauge_preco_mediano_bairro.labels(
            bairro=bairro_norm, zona=zona_norm
        ).set(preco_mediano)
        self._gauge_amostras_bairro.labels(
            bairro=bairro_norm, zona=zona_norm
        ).set(float(total_amostras))
