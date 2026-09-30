from typing import Protocol, runtime_checkable


@runtime_checkable
class ContratoTelemetria(Protocol):
    def registrar_inferencia(
        self, duracao_segundos: float, zona: str, valor_previsto: float
    ) -> None: ...

    def atualizar_metricas_modelo(self, rmse: float, mae: float, r2: float) -> None: ...

    def registrar_drift_dados(
        self,
        psi_predicoes: float,
        psi_area: float,
        ks_stat_area: float,
        ks_pval_area: float,
        wasserstein_area: float,
        status_geral: int,
    ) -> None: ...

    def registrar_drift_feature(self, feature: str, psi: float) -> None: ...

    def registrar_desvio_preco_zona(self, zona: str, desvio_percentual: float) -> None: ...

    def registrar_testes_estatisticos(
        self,
        friedman_chi2: float,
        friedman_pvalor: float,
        friedman_significativo: bool,
        nemenyi_cd: float,
        ranks_modelos: dict[str, float],
        comparacoes_nemenyi: list[tuple[str, str, float, bool]],
        resultados_shapiro: dict[str, tuple[float, bool]],
    ) -> None: ...

    # --- Métricas de Pipeline ---
    def registrar_inicio_etapa(self, nome_etapa: str, indice: int, total: int) -> None: ...

    def registrar_fim_etapa(
        self, nome_etapa: str, indice: int, total: int, duracao_segundos: float, sucesso: bool
    ) -> None: ...

    def registrar_pipeline_concluido(
        self, duracao_total_segundos: float, total_etapas: int, etapas_com_falha: int
    ) -> None: ...

    # --- Métricas Estendidas do Modelo ---
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
    ) -> None: ...

    # --- Métricas de Negócio Hierárquico ---
    def registrar_metricas_negocio(
        self,
        preco_mediano_global: float,
        preco_medio_global: float,
        preco_m2_mediano_global: float,
        total_imoveis_dataset: int,
    ) -> None: ...

    def registrar_metricas_negocio_zona(
        self, zona: str, preco_mediano: float, preco_m2_mediano: float, total_amostras: int
    ) -> None: ...

    def registrar_metricas_negocio_bairro(
        self, bairro: str, zona: str, preco_mediano: float, total_amostras: int
    ) -> None: ...
