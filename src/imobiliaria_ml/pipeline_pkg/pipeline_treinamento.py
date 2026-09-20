"""Template Method para execução completa do pipeline de Machine Learning."""

import logging
from pathlib import Path
import warnings
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ..carregamento_pkg.carregador_csv import CarregadorCsv
from ..carregamento_pkg.fabrica_carregadores import FabricaCarregadores
from ..configuracao_pkg.configuracao_projeto import ConfiguracaoProjeto
from ..eda_pkg.gerador_relatorio_eda import GeradorRelatorioEda
from ..ensemble_pkg.calculador_pesos_ranking import CalculadorPesosRanking
from ..ensemble_pkg.calculador_pesos_rmse import CalculadorPesosRmse
from ..ensemble_pkg.fabrica_ensemble import FabricaEnsemble
from ..ensemble_pkg.seletor_modelos_ensemble import SeletorModelosEnsemble
from ..enums_pkg.tipo_carregador import TipoCarregador
from ..enums_pkg.tipo_ensemble import TipoEnsemble
from ..enums_pkg.tipo_evento import TipoEvento
from ..enums_pkg.tipo_modelo import TipoModelo
from ..explicabilidade_pkg.explicador_importancias import ExplicadorImportancias
from ..explicabilidade_pkg.extrator_coeficientes import ExtratorCoeficientes
from ..explicabilidade_pkg.interpretador_modelo import InterpretadorModelo
from ..explicabilidade_pkg.resultado_explicabilidade import ResultadoExplicabilidade
from ..mlflow_pkg.carga_evento import CargaEvento
from ..mlflow_pkg.observador_mlflow import ObservadorMlflow
from ..mlflow_pkg.registrador_modelo import RegistradorModelo
from ..mlflow_pkg.sujeito_observavel import SujeitoObservavel
from ..modelos_pkg.fabrica_modelos import FabricaModelos
from ..modelos_pkg.parametros_modelo import ParametrosModelo
from ..modelos_pkg.regressor_protocol import RegressorProtocol
from ..negocio_pkg.calculador_metricas_imobiliaria import CalculadorMetricasImobiliaria
from ..negocio_pkg.metricas_negocio_resultado import MetricasNegocioResultado
from ..preprocessamento_pkg.fabrica_escalonadores import FabricaEscalonadores
from ..preprocessamento_pkg.pre_processador import PreProcessador
from ..preprocessamento_pkg.tratador_categorico import TratadorCategorico
from ..qualidade_pkg.validador_esquema import ValidadorEsquema
from ..qualidade_pkg.validador_qualidade import ValidadorQualidade
from ..selecao_pkg.seletor_hiperparametros import SeletorHiperparametros
from ..selecao_pkg.explicador_hiperparametros import ExplicadorHiperparametros
from ..validacao_pkg.analisador_aprendizado import AnalisadorAprendizado
from ..validacao_pkg.calculador_metricas import CalculadorMetricas
from ..validacao_pkg.grupo_elegivel import GrupoElegivel
from ..validacao_pkg.metricas_regressao import MetricasRegressao
from ..validacao_pkg.resultado_friedman import ResultadoFriedman
from ..validacao_pkg.resultado_nemenyi import ResultadoNemenyi
from ..validacao_pkg.resultado_validacao_cruzada import ResultadoValidacaoCruzada
from ..validacao_pkg.seletor_campeao import SeletorCampeao
from ..validacao_pkg.teste_friedman import TesteFriedman
from ..validacao_pkg.teste_nemenyi import TesteNemenyi
from ..validacao_pkg.validador_repeated_kfold import ValidadorRepeatedKFold
from .candidato_modelo import CandidatoModelo
from .resultado_treinamento import ResultadoTreinamento

logger = logging.getLogger(__name__)


class PipelineTreinamento(SujeitoObservavel):
    """Template Method que comanda o ciclo oficial de Machine Learning."""

    COLUNAS_NUMERICAS: tuple[str, ...] = ("Quartos", "Banheiros", "Vagas", "Metragem")
    COLUNAS_CATEGORICAS: tuple[str, ...] = ("Zona",)

    def __init__(
        self,
        configuracao: ConfiguracaoProjeto | None = None,
        modelos_para_executar: list[TipoModelo] | None = None,
    ) -> None:
        super().__init__()
        self._config = configuracao or ConfiguracaoProjeto()
        self._modelos_selecionados = modelos_para_executar
        self._fabrica_carregadores = FabricaCarregadores()
        self._validador_esquema = ValidadorEsquema()
        self._validador_qualidade = ValidadorQualidade()
        self._gerador_eda = GeradorRelatorioEda()
        self._fabrica_escalonadores = FabricaEscalonadores()
        self._fabrica_modelos = FabricaModelos()
        self._seletor_hiperparametros = SeletorHiperparametros(
            cv_splits=self._config.n_splits
        )
        self._explicador_hiper = ExplicadorHiperparametros()
        self._validador_cv = ValidadorRepeatedKFold(
            n_splits=self._config.n_splits,
            n_repeats=self._config.n_repeats,
            random_state=self._config.seed,
        )
        self._teste_friedman = TesteFriedman(alpha=self._config.alpha_friedman)
        self._teste_nemenyi = TesteNemenyi(alpha=self._config.alpha_nemenyi)
        self._seletor_campeao = SeletorCampeao()
        self._fabrica_ensemble = FabricaEnsemble()
        self._seletor_ensemble = SeletorModelosEnsemble()
        self._calculador_pesos_rmse = CalculadorPesosRmse()
        self._calculador_pesos_ranking = CalculadorPesosRanking()
        self._calculador_metricas = CalculadorMetricas()
        self._analisador_aprendizado = AnalisadorAprendizado()
        self._extrator_coef = ExtratorCoeficientes()
        self._interpretador = InterpretadorModelo()
        self._explicador_imp = ExplicadorImportancias()
        self._calculador_negocio = CalculadorMetricasImobiliaria()
        self._registrador = RegistradorModelo(
            nome_modelo=self._config.mlflow_nome_modelo,
            usar_votacao=self._config.usar_votacao,
        )

        # Adiciona automaticamente o observador oficial do MLflow com tags de governança
        tags_governanca: dict[str, str] = {
            "alvo": self._config.alvo,
            "seed": str(self._config.seed),
            "metrica_primaria": self._config.metrica_primaria.value,
        }
        self.adicionar_observador(
            ObservadorMlflow(
                nome_experimento=self._config.mlflow_experimento,
                tags_padrao=tags_governanca,
                usar_votacao=self._config.usar_votacao,
            )
        )

    def carregar_dados(self, origem: str | Path, tipo: TipoCarregador = TipoCarregador.CSV) -> pd.DataFrame:
        carregador = self._fabrica_carregadores.criar(tipo)
        return carregador.carregar_dados(origem)

    def validar_dados(self, dados: pd.DataFrame) -> None:
        res_esquema = self._validador_esquema.validar(dados, exigir_alvo=True)
        if not res_esquema.valido:
            raise ValueError(f"Falha na validação de esquema: {res_esquema.mensagens_erro}")

        res_qualidade = self._validador_qualidade.validar(dados)
        if not res_qualidade.valido:
            raise ValueError(f"Falha na qualidade de dados: {res_qualidade.mensagens_erro}")

    def executar_eda(self, dados: pd.DataFrame) -> None:
        cols_num = list(self.COLUNAS_NUMERICAS) + [self._config.alvo]
        cols_cat = list(self.COLUNAS_CATEGORICAS)
        estatisticas, df_descritivo, df_categorias, figuras = (
            self._gerador_eda.gerar_relatorio_completo(
                dados=dados,
                colunas_numericas=cols_num,
                colunas_categoricas=cols_cat,
                coluna_alvo=self._config.alvo,
            )
        )
        payload: dict[
            str,
            str
            | int
            | float
            | bool
            | pd.DataFrame
            | Figure
            | dict[str, float | int | str | bool | None]
            | list[str]
            | None,
        ] = {
            "estatisticas_descritivas": df_descritivo,
            "categorias": df_categorias,
            "correlacao_pearson": figuras.get("correlacao_pearson"),
            "correlacao_spearman": figuras.get("correlacao_spearman"),
            "distribuicao_alvo": figuras.get("distribuicao_alvo"),
            "boxplot_zona_valor": figuras.get("boxplot_zona_valor"),
            "boxplot_zona_m2": figuras.get("boxplot_zona_m2"),
            "perfil_imobiliario_zona": figuras.get("perfil_imobiliario_zona"),
            "tabela_zonas": estatisticas.tabela_zonas_df,
            "diagnostico_zonas": estatisticas.diagnostico_zona,
            "interpretacao_zonas": estatisticas.resumo_zonas_md,
            "relatorio_negocio_eda": estatisticas.relatorio_negocio_md,
        }
        if estatisticas.diagnostico_zona:
            logger.info("   * Zona Mais Valorizada: %s", estatisticas.diagnostico_zona.get("zona_mais_valorizada", ""))
            logger.info("   * Zona Mais Acessível: %s", estatisticas.diagnostico_zona.get("zona_mais_acessivel", ""))
            logger.info("   * Gradiente Espacial: %s", estatisticas.diagnostico_zona.get("fator_gradiente_espacial", ""))
            logger.info("   * Maior Volume Amostral: %s", estatisticas.diagnostico_zona.get("zona_maior_liquidez", ""))

        self.notificar(TipoEvento.EDA_FINALIZADA, CargaEvento(valores=payload))

    def separar_holdout(
        self, dados: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return train_test_split(  # type: ignore[return-value]
            dados, test_size=self._config.test_size, random_state=self._config.seed
        )

    def extrair_features_alvo(
        self, dados: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Separa as variáveis preditoras (features) e a variável alvo."""
        colunas = list(self.COLUNAS_NUMERICAS) + list(self.COLUNAS_CATEGORICAS)
        X = dados[colunas]
        y = dados[self._config.alvo]
        return X, y

    def criar_candidatos(self) -> list[TipoModelo]:
        """Cria e retorna a lista de tipos de modelos candidatos a serem otimizados."""
        if self._modelos_selecionados:
            return self._modelos_selecionados
        return [
            TipoModelo.REGRESSAO_LINEAR,
            TipoModelo.REGRESSAO_MULTIPLA,
            TipoModelo.REGRESSAO_POLINOMIAL,
            TipoModelo.RIDGE,
            TipoModelo.LASSO,
            TipoModelo.ELASTIC_NET,
            TipoModelo.ARVORE_DECISAO,
            TipoModelo.RANDOM_FOREST,
            TipoModelo.GRADIENT_BOOSTING,
            TipoModelo.SVR,
            TipoModelo.REDE_NEURAL,
            TipoModelo.XGBOOST,
            TipoModelo.LIGHTGBM,
            TipoModelo.CATBOOST,
        ]

    def _obter_modelos_candidatos(self) -> list[TipoModelo]:
        """Alias para criar_candidatos mantido para compatibilidade interna."""
        return self.criar_candidatos()

    def _criar_pipeline_modelo(self, tipo: TipoModelo) -> Pipeline:
        estrategia = self._fabrica_modelos.criar(tipo)
        scaler = self._fabrica_escalonadores.criar(estrategia.obter_escalonador_padrao())
        pre = PreProcessador(
            estrategia_escalonamento=scaler,
            tratador_categorico=TratadorCategorico(),
        ).construir_pipeline(
            colunas_numericas=list(self.COLUNAS_NUMERICAS),
            colunas_categoricas=list(self.COLUNAS_CATEGORICAS),
        )
        modelo_base = estrategia.criar_modelo()
        return Pipeline(steps=[("pre", pre), ("modelo", modelo_base)])

    def executar_grid_search(
        self,
        tipos_candidatos: list[TipoModelo],
        X_dev: pd.DataFrame,
        y_dev: pd.Series,
    ) -> tuple[dict[str, Pipeline], dict[str, CandidatoModelo]]:
        """Executa a otimização de hiperparâmetros (GridSearchCV) para os modelos candidatos."""
        logger.info("[Passo 5/%d] Otimizando hiperparâmetros (GridSearchCV com %d splits) para %d modelos candidatos...", self._config.total_passos, self._config.n_splits, len(tipos_candidatos))
        pipelines_otimizadas: dict[str, Pipeline] = {}
        candidatos_modelos: dict[str, CandidatoModelo] = {}

        for idx, tipo in enumerate(tipos_candidatos, start=1):
            logger.info(" [%d/%d] Otimizando modelo: %s...", idx, len(tipos_candidatos), tipo.value)
            estrategia = self._fabrica_modelos.criar(tipo)
            pipeline_base = self._criar_pipeline_modelo(tipo)

            self.notificar(
                TipoEvento.GRIDSEARCH_INICIADO,
                CargaEvento(valores={"nome_modelo": tipo.value}),
            )

            res_grid = self._seletor_hiperparametros.otimizar(
                estrategia=estrategia,
                pipeline_base=pipeline_base,
                X_treino=X_dev,
                y_treino=y_dev,
            )

            logger.info("   -> [%s] Concluído | Melhor RMSE (CV): R$ %s | Melhores parâmetros: %s", tipo.value, f"{res_grid.melhor_score_rmse:,.2f}", res_grid.melhores_parametros)

            # Gera relatório de negócio dos parâmetros selecionados
            _, relatorio_params_md = self._explicador_hiper.gerar_relatorio_negocio(
                nome_modelo=tipo.value,
                melhores_parametros=res_grid.melhores_parametros,
            )

            pipeline_otimizada = clone(pipeline_base)
            pipeline_otimizada.set_params(**res_grid.melhores_parametros)
            pipeline_otimizada.fit(X_dev, y_dev)
            pipelines_otimizadas[tipo.value] = pipeline_otimizada

            # Extrai equação analítica/matemática do modelo candidato otimizado
            equacao_candidato_txt = self._extrator_coef.gerar_equacao_txt(
                pipeline_ajustada=pipeline_otimizada,
                nome_modelo=tipo.value,
            )

            self.notificar(
                TipoEvento.GRIDSEARCH_FINALIZADO,
                CargaEvento(
                    valores={
                        "nome_modelo": tipo.value,
                        "melhores_parametros": res_grid.melhores_parametros,
                        "melhor_score": res_grid.melhor_score_rmse,
                        "tabela_cv_results": res_grid.tabela_cv_results,
                        "relatorio_params_negocio_md": relatorio_params_md,
                        "equacao_texto": equacao_candidato_txt,
                    }
                ),
            )

            candidatos_modelos[tipo.value] = CandidatoModelo(
                tipo_modelo=tipo,
                nome=tipo.value,
                pipeline=pipeline_otimizada,
                melhores_parametros=res_grid.melhores_parametros,
                rmse_cv=res_grid.melhor_score_rmse,
            )

        return pipelines_otimizadas, candidatos_modelos

    def extrair_explicabilidade_candidatos(
        self,
        pipelines_otimizadas: dict[str, Pipeline],
        X_dev: pd.DataFrame,
        y_dev: pd.Series,
    ) -> None:
        """Treina cada pipeline candidata (clone) em 100% do dev e extrai equação ou importâncias."""
        logger.info(" -> Extraindo explicabilidade de %d modelos candidatos com melhores parâmetros...", len(pipelines_otimizadas))
        for nome, pipeline_base in pipelines_otimizadas.items():
            try:
                pipeline_ajustada = clone(pipeline_base)
                pipeline_ajustada.fit(X_dev, y_dev)
                estimador = pipeline_ajustada.named_steps.get("modelo")

                payload_exp: dict[
                    str,
                    str
                    | int
                    | float
                    | bool
                    | pd.DataFrame
                    | dict[str, float | int | str | bool | None]
                    | list[str]
                    | None,
                ] = {"nome_modelo": nome}

                # Desce para dentro de pipelines aninhadas (ex: Regressão Polinomial)
                if isinstance(estimador, Pipeline):
                    estimador_final_check = estimador.named_steps.get("linear") or estimador.steps[-1][1]
                else:
                    estimador_final_check = estimador

                equacao_candidato_txt = self._extrator_coef.gerar_equacao_txt(
                    pipeline_ajustada=pipeline_ajustada,
                    nome_modelo=nome,
                )

                if hasattr(estimador_final_check, "coef_"):
                    df_coef, intercepto, equacao = self._extrator_coef.extrair(pipeline_ajustada)
                    md_interp = self._interpretador.gerar_interpretacao_markdown(df_coef, intercepto)
                    payload_exp.update(
                        {
                            "tipo": "coeficientes",
                            "equacao_texto": equacao_candidato_txt,
                            "equacao_linha": equacao,
                            "intercepto": intercepto,
                            "tabela_coeficientes": df_coef,
                            "interpretacao_texto": md_interp,
                        }
                    )
                    logger.info("   -> [%s] Equação matemática extraída.", nome)
                else:
                    df_imp = self._explicador_imp.calcular_importancias(
                        pipeline_ajustada=pipeline_ajustada,
                        X_val=X_dev,
                        y_val=y_dev,
                    )
                    payload_exp.update(
                        {
                            "tipo": "importancias",
                            "equacao_texto": equacao_candidato_txt,
                            "tabela_importancias": df_imp,
                        }
                    )
                    logger.info("   -> [%s] Importâncias (%d features) e formulação analítica extraídas.", nome, len(df_imp))

                self.notificar(
                    TipoEvento.EXPLICABILIDADE_CANDIDATO_GERADA,
                    CargaEvento(valores=payload_exp),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("   -> [%s] Falha ao extrair explicabilidade: %s", nome, exc)

    def validar_repeated_kfold(
        self,
        pipelines_otimizadas: dict[str, Pipeline],
        X_dev: pd.DataFrame,
        y_dev: pd.Series,
    ) -> ResultadoValidacaoCruzada:
        """Executa a validação cruzada robusta com 30 repetições compartilhadas."""
        logger.info("[Passo 6/%d] Executando validação cruzada RepeatedKFold (%d splits x %d repetições = %d dobras por modelo)...", self._config.total_passos, self._config.n_splits, self._config.n_repeats, self._config.n_splits * self._config.n_repeats)
        res_cv = self._validador_cv.validar_modelos(
            modelos_pipelines=pipelines_otimizadas,
            X_treino=X_dev,
            y_treino=y_dev,
        )
        logger.info(" -> Validação cruzada RepeatedKFold concluída. Matriz 30 repetições gerada com sucesso.")
        self.notificar(
            TipoEvento.VALIDACAO_FINALIZADA,
            CargaEvento(
                valores={
                    "tabela_folds": res_cv.tabela_folds,
                    "matriz_repeticoes": res_cv.matriz_repeticoes,
                    "resumo_modelos": res_cv.resumo_modelos,
                }
            ),
        )
        return res_cv

    def executar_testes_estatisticos(
        self,
        res_cv: ResultadoValidacaoCruzada,
    ) -> tuple[ResultadoFriedman, ResultadoNemenyi]:
        """Aplica os testes estatísticos comparativos de Friedman e Nemenyi."""
        logger.info("[Passo 7/%d] Aplicando testes estatísticos pós-validação cruzada (Friedman & Nemenyi)...", self._config.total_passos)
        res_friedman = self._teste_friedman.executar(res_cv.matriz_repeticoes)
        logger.info(" -> Teste de Friedman: Estatística Chi² = %.4f | p-valor = %.4e (alpha = %.2f)", res_friedman.estatistica, res_friedman.p_valor, res_friedman.alpha)
        if res_friedman.significativo:
            logger.info(" -> Diferença estatística significativa detectada (p < alpha).")
        else:
            logger.info(" -> Não foi detectada diferença estatisticamente significativa entre os modelos.")

        self.notificar(
            TipoEvento.FRIEDMAN_FINALIZADO,
            CargaEvento(
                valores={
                    "friedman": {
                        "estatistica": res_friedman.estatistica,
                        "p_valor": res_friedman.p_valor,
                        "alpha": res_friedman.alpha,
                        "significativo": res_friedman.significativo,
                    },
                    "p_valor": res_friedman.p_valor,
                    "estatistica": res_friedman.estatistica,
                }
            ),
        )

        res_nemenyi = self._teste_nemenyi.executar(res_cv.matriz_repeticoes, res_friedman)
        if res_nemenyi.executado:
            logger.info(" -> Pós-teste de Nemenyi executado: Distância Crítica (CD) = %.4f", res_nemenyi.distancia_critica)
            self.notificar(
                TipoEvento.NEMENYI_FINALIZADO,
                CargaEvento(valores={"matriz_p_valores": res_nemenyi.matriz_p_valores}),
            )
        else:
            logger.info(" -> Pós-teste de Nemenyi não necessário ou tratamentos insuficientes.")

        return res_friedman, res_nemenyi

    def selecionar_grupo_elegivel(
        self,
        res_cv: ResultadoValidacaoCruzada,
        res_friedman: ResultadoFriedman,
        res_nemenyi: ResultadoNemenyi,
    ) -> GrupoElegivel:
        """Determina o grupo de modelos estatisticamente indistinguíveis do melhor."""
        logger.info("[Passo 8/%d] Delimitando o Grupo Estatisticamente Elegível...", self._config.total_passos)
        grupo = self._seletor_campeao.selecionar_grupo(res_cv, res_friedman, res_nemenyi)
        logger.info(" -> Melhor modelo individual: [%s] (RMSE médio CV: R$ %s)", grupo.melhor_modelo, f"{grupo.rmse_medios.get(grupo.melhor_modelo, 0.0):,.2f}")
        logger.info(" -> Modelos elegíveis (%d): %s", len(grupo.modelos_elegiveis), ", ".join(grupo.modelos_elegiveis))
        return grupo

    def selecionar_modelo_individual(
        self,
        pipelines_otimizadas: dict[str, Pipeline],
        grupo: GrupoElegivel,
    ) -> tuple[str, Pipeline]:
        """Seleciona o melhor modelo individual dentro do grupo elegível."""
        melhor_individual_nome = grupo.melhor_modelo
        pipeline_individual = pipelines_otimizadas[melhor_individual_nome]
        return melhor_individual_nome, pipeline_individual

    def avaliar_ensembles(
        self,
        pipelines_otimizadas: dict[str, Pipeline],
        grupo: GrupoElegivel,
        X_dev: pd.DataFrame,
        y_dev: pd.Series,
    ) -> tuple[str, Pipeline, MetricasRegressao]:
        """Constrói e avalia o ensemble formado exclusivamente pelos modelos elegíveis."""
        logger.info("[Passo 9/%d] Avaliando estratégias de Ensemble (usar_votacao = True)...", self._config.total_passos)
        modelos_elegiveis = self._seletor_ensemble.filtrar_elegiveis(
            pipelines_otimizadas, grupo
        )
        pesos_rmse = self._calculador_pesos_rmse.calcular(grupo.rmse_medios)
        logger.info(" -> Modelos que compõem o Ensemble: %s", [m[0] for m in modelos_elegiveis])
        logger.info(" -> Pesos atribuídos por RMSE: %s", {k: f"{v:.4f}" for k, v in pesos_rmse.items()})

        estrategia_voting = self._fabrica_ensemble.criar(
            TipoEnsemble.VOTING_PONDERADO_RMSE, pesos_por_modelo=pesos_rmse
        )
        ensemble_estimador = estrategia_voting.criar_estimador(
            [(nome, pipe.named_steps["modelo"]) for nome, pipe in modelos_elegiveis]
        )

        scaler_ens = self._fabrica_escalonadores.criar(
            self._fabrica_modelos.criar(TipoModelo.RIDGE).obter_escalonador_padrao()
        )
        pre_ens = PreProcessador(
            estrategia_escalonamento=scaler_ens
        ).construir_pipeline(
            colunas_numericas=list(self.COLUNAS_NUMERICAS),
            colunas_categoricas=list(self.COLUNAS_CATEGORICAS),
        )
        pipeline_ensemble = Pipeline(
            steps=[("pre", pre_ens), ("modelo", ensemble_estimador)]
        )

        pipeline_ensemble.fit(X_dev, y_dev)
        preds_ens_dev = pipeline_ensemble.predict(X_dev)
        metricas_ens_dev = self._calculador_metricas.calcular(y_dev, preds_ens_dev)

        self.notificar(
            TipoEvento.ENSEMBLE_FINALIZADO,
            CargaEvento(
                valores={
                    "usado": True,
                    "estrategia": TipoEnsemble.VOTING_PONDERADO_RMSE.value,
                    "pesos": pesos_rmse,
                    "metricas": {
                        "rmse": metricas_ens_dev.rmse,
                        "mae": metricas_ens_dev.mae,
                        "mse": metricas_ens_dev.mse,
                        "r2": metricas_ens_dev.r2,
                        "medae": metricas_ens_dev.medae,
                    },
                }
            ),
        )

        return "ensemble_voting", pipeline_ensemble, metricas_ens_dev

    def comparar_candidatos(
        self,
        nome_individual: str,
        pipeline_individual: Pipeline,
        ensemble_resultado: tuple[str, Pipeline, MetricasRegressao],
        grupo: GrupoElegivel,
    ) -> tuple[str, Pipeline]:
        """Compara o ensemble contra o melhor modelo individual para definir o campeão."""
        nome_ens, pipe_ens, metricas_ens = ensemble_resultado
        rmse_melhor_ind = grupo.rmse_medios.get(nome_individual, float("inf"))
        if metricas_ens.rmse < rmse_melhor_ind:
            logger.info(" -> Ensemble SUPEROU o melhor individual! (RMSE Ensemble: R$ %s vs Individual: R$ %s)", f"{metricas_ens.rmse:,.2f}", f"{rmse_melhor_ind:,.2f}")
            return nome_ens, pipe_ens
        else:
            logger.info(" -> Melhor individual [%s] superou o Ensemble (R$ %s vs R$ %s). Mantido individual.", nome_individual, f"{rmse_melhor_ind:,.2f}", f"{metricas_ens.rmse:,.2f}")
            return nome_individual, pipeline_individual

    def treinar_final(
        self,
        pipeline_campea: Pipeline,
        X_dev: pd.DataFrame,
        y_dev: pd.Series,
        nome_campeao: str,
    ) -> Pipeline:
        """Treina a pipeline campeã em 100% da base de desenvolvimento."""
        logger.info("[Passo 10/%d] Realizando treinamento final do campeão [%s] em 100%% da base de desenvolvimento (%d registros)...", self._config.total_passos, nome_campeao, len(X_dev))
        pipeline_campea.fit(X_dev, y_dev)
        logger.info(" -> Ajuste final do modelo campeão concluído com sucesso.")
        return pipeline_campea

    def avaliar_holdout(
        self,
        pipeline_campea: Pipeline,
        X_holdout: pd.DataFrame,
        y_holdout: pd.Series,
        nome_campeao: str,
        rmse_cv: float | None = None,
    ) -> tuple[np.ndarray, MetricasRegressao]:
        """Avalia o modelo campeão no conjunto de holdout inédito."""
        logger.info("[Passo 11/%d] Avaliando campeão no Holdout Final inédito (%d registros) e calculando métricas imobiliárias...", self._config.total_passos, len(X_holdout))
        y_pred_holdout = np.asarray(pipeline_campea.predict(X_holdout), dtype=float)
        metricas_holdout = self._calculador_metricas.calcular(y_holdout, y_pred_holdout)

        logger.info("--------------------------------------------------------------------------------")
        logger.info(" DESEMPENHO DO MODELO CAMPEÃO [%s] NO HOLDOUT FINAL:", nome_campeao)
        logger.info("   * RMSE : R$ %s", f"{metricas_holdout.rmse:,.2f}")
        logger.info("   * MAE  : R$ %s", f"{metricas_holdout.mae:,.2f}")
        logger.info("   * R²   : %.4f", metricas_holdout.r2)
        logger.info("   * MAPE : %.2f%%", metricas_holdout.mape)
        logger.info("--------------------------------------------------------------------------------")

        valores_campeao: dict[str, object] = {
            "nome_campeao": nome_campeao,
            "metricas_holdout": {
                "rmse": metricas_holdout.rmse,
                "mae": metricas_holdout.mae,
                "mse": metricas_holdout.mse,
                "r2": metricas_holdout.r2,
                "medae": metricas_holdout.medae,
            },
        }
        if rmse_cv is not None:
            valores_campeao["rmse_cv"] = float(rmse_cv)

        self.notificar(
            TipoEvento.MODELO_CAMPEAO,
            CargaEvento(valores=valores_campeao),
        )
        return y_pred_holdout, metricas_holdout

    def gerar_curva_aprendizado(
        self,
        pipeline_campea: Pipeline,
        X_dev: pd.DataFrame,
        y_dev: pd.Series,
    ) -> None:
        """Gera a curva de aprendizado para diagnóstico de Underfitting e Overfitting."""
        logger.info(" -> Gerando Curva de Aprendizado e diagnóstico de Underfitting/Overfitting...")
        figura_curva, df_curva, diag_info = self._analisador_aprendizado.gerar_curva(
            modelo_pipeline=pipeline_campea,
            X_treino=X_dev,
            y_treino=y_dev,
            cv=self._config.n_splits,
        )
        logger.info(" -> Diagnóstico da Curva de Aprendizado: %s", diag_info)
        self.notificar(
            TipoEvento.CURVA_APRENDIZADO_GERADA,
            CargaEvento(
                valores={
                    "figura_curva": figura_curva,
                    "tabela_curva": df_curva,
                    "diagnostico": diag_info,
                }
            ),
        )

    def extrair_explicabilidade(
        self,
        pipeline_campea: Pipeline,
        X_holdout: pd.DataFrame,
        y_holdout: pd.Series,
        nome_campeao: str,
    ) -> ResultadoExplicabilidade | None:
        """Extrai coeficientes e equações ou importância das variáveis via permutação."""
        logger.info(" -> Extraindo explicabilidade e importância das variáveis do campeão...")
        resultado_explicabilidade: ResultadoExplicabilidade | None = None
        estimador_interno = pipeline_campea.named_steps.get("modelo")
        # Desce para dentro de pipelines aninhadas (ex: Regressão Polinomial)
        if isinstance(estimador_interno, Pipeline):
            estimador_final_check = estimador_interno.named_steps.get("linear") or estimador_interno.steps[-1][1]
        else:
            estimador_final_check = estimador_interno
        equacao_campeao_txt = self._extrator_coef.gerar_equacao_txt(
            pipeline_ajustada=pipeline_campea,
            nome_modelo=nome_campeao,
        )

        if hasattr(estimador_final_check, "coef_"):
            df_coef, intercepto, equacao = self._extrator_coef.extrair(pipeline_campea)
            md_interp = self._interpretador.gerar_interpretacao_markdown(df_coef, intercepto)
            resultado_explicabilidade = ResultadoExplicabilidade(
                tipo_modelo=nome_campeao,
                intercepto=intercepto,
                tabela_coeficientes=df_coef,
                equacao_texto=equacao_campeao_txt,
                interpretacao_texto=md_interp,
            )
            logger.info(" -> Equação matemática do campeão extraída com sucesso.")
            self.notificar(
                TipoEvento.EXPLICABILIDADE_GERADA,
                CargaEvento(
                    valores={
                        "nome_campeao": nome_campeao,
                        "tipo": "coeficientes",
                        "equacao_texto": equacao_campeao_txt,
                        "equacao_linha": equacao,
                        "intercepto": intercepto,
                        "tabela_coeficientes": df_coef,
                        "interpretacao_texto": md_interp,
                    }
                ),
            )
        else:
            df_imp = self._explicador_imp.calcular_importancias(
                pipeline_ajustada=pipeline_campea,
                X_val=X_holdout,
                y_val=y_holdout,
            )
            resultado_explicabilidade = ResultadoExplicabilidade(
                tipo_modelo=nome_campeao,
                equacao_texto=equacao_campeao_txt,
                tabela_importancias=df_imp,
            )
            logger.info(" -> Importâncias e formulação matemática do campeão extraídas com sucesso.")
            self.notificar(
                TipoEvento.EXPLICABILIDADE_GERADA,
                CargaEvento(
                    valores={
                        "nome_campeao": nome_campeao,
                        "tipo": "importancias",
                        "equacao_texto": equacao_campeao_txt,
                        "tabela_importancias": df_imp,
                    }
                ),
            )
        return resultado_explicabilidade

    def calcular_metricas_negocio(
        self,
        X_holdout: pd.DataFrame,
        y_holdout: pd.Series,
        y_pred_holdout: np.ndarray,
        nome_campeao: str = "",
    ) -> MetricasNegocioResultado:
        """Calcula métricas de negócio imobiliário (coberturas de margem e desconto seguro)."""
        logger.info(" -> Consolidando métricas de negócio imobiliário (coberturas e desconto seguro)...")
        metricas_negocio = self._calculador_negocio.calcular_metricas(
            X_holdout=X_holdout,
            y_real=y_holdout,
            y_pred=y_pred_holdout,
        )
        logger.info("   * Cobertura de Erro até  5%%: %.2f%%", metricas_negocio.cobertura_5 * 100)
        logger.info("   * Cobertura de Erro até 10%%: %.2f%%", metricas_negocio.cobertura_10 * 100)
        logger.info("   * Cobertura de Erro até 15%%: %.2f%%", metricas_negocio.cobertura_15 * 100)
        logger.info("   * Desconto Seguro Recomendado: %.2f%%", metricas_negocio.desconto_seguro_recomendado)
        logger.info("   * Margem de Negociação Estimada: R$ %s", f"{metricas_negocio.margem_negociacao_estimada:,.2f}")
        logger.info("   * Receita Potencial Perdida: R$ %s", f"{metricas_negocio.receita_potencial_perdida:,.2f}")

        self.notificar(
            TipoEvento.METRICAS_NEGOCIO_FINALIZADAS,
            CargaEvento(
                valores={
                    "nome_campeao": nome_campeao,
                    "metricas_consolidadas": {
                        "cobertura_5": metricas_negocio.cobertura_5,
                        "cobertura_10": metricas_negocio.cobertura_10,
                        "cobertura_15": metricas_negocio.cobertura_15,
                        "mae_reais": metricas_negocio.mae_reais,
                        "vies_medio": metricas_negocio.vies_medio,
                        "desconto_seguro": metricas_negocio.desconto_seguro_recomendado,
                        "desconto_seguro_recomendado": metricas_negocio.desconto_seguro_recomendado,
                        "risco_subprecificacao": metricas_negocio.risco_subprecificacao,
                        "risco_superprecificacao": metricas_negocio.risco_superprecificacao,
                        "margem_negociacao_estimada": metricas_negocio.margem_negociacao_estimada,
                        "receita_potencial_perdida": metricas_negocio.receita_potencial_perdida,
                    },
                    "tabela_erro_faixa": metricas_negocio.tabela_erro_por_faixa,
                    "tabela_erro_zona": metricas_negocio.tabela_erro_por_zona,
                    "tabela_cobertura_tolerancia": metricas_negocio.tabela_cobertura_tolerancia,
                }
            ),
        )
        return metricas_negocio

    def registrar_modelo(
        self,
        pipeline_campea: Pipeline,
        metricas_holdout: MetricasRegressao,
        nome_campeao: str = "",
        metricas_negocio: MetricasNegocioResultado | None = None,
    ) -> str:
        """Registra o modelo campeão no Model Registry do MLflow com alias @champion."""
        logger.info("[Passo 12/%d] Registrando modelo campeão encapsulado em PyFunc no MLflow Model Registry...", self._config.total_passos)
        uri_registro = self._registrador.registrar_modelo_campeao(
            pipeline_campea=pipeline_campea,
            metricas={
                "rmse": metricas_holdout.rmse,
                "mae": metricas_holdout.mae,
                "r2": metricas_holdout.r2,
            },
            desconto_maximo=self._config.desconto_maximo_com_aprovacao,
            nome_modelo_concreto=nome_campeao,
            metricas_negocio=metricas_negocio,
        )
        logger.info(" -> Modelo registrado no MLflow com alias @champion: %s", uri_registro)
        return uri_registro

    def executar(self, caminho_dados: str | Path) -> ResultadoTreinamento[RegressorProtocol]:
        """Template Method que orquestra o ciclo oficial completo de Machine Learning."""
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

        logger.info("================================================================================")
        logger.info("INICIANDO PIPELINE OFICIAL DE MACHINE LEARNING — PREVISÃO DE PREÇO DE IMÓVEIS")
        logger.info("Variável Alvo: %s | Semente Aleatória: %d", self._config.alvo, self._config.seed)
        logger.info("================================================================================")

        # 1. Carregar dados
        logger.info("[Passo 1/%d] Carregando base de dados a partir de: %s", self._config.total_passos, caminho_dados)
        dados = self.carregar_dados(caminho_dados)
        logger.info(" -> Base carregada com sucesso: %d registros e %d colunas.", len(dados), len(dados.columns))

        # 2. Validar esquema e qualidade
        logger.info("[Passo 2/%d] Validando conformidade de esquema e integridade de qualidade...", self._config.total_passos)
        self.validar_dados(dados)
        logger.info(" -> Validação concluída: colunas obrigatórias presentes, sem duplicatas e dados válidos.")

        # 3. EDA obrigatória antes do treino
        logger.info("[Passo 3/%d] Executando Análise Exploratória de Dados (EDA) obrigatória...", self._config.total_passos)
        self.executar_eda(dados)
        logger.info(" -> EDA finalizada: correlações, estatísticas e gráficos de distribuição gerados em memória.")

        # 4. Separação desenvolvimento / holdout final
        pct_teste = self._config.test_size * 100
        pct_dev = (1.0 - self._config.test_size) * 100
        fmt_dev = f"{pct_dev:.0f}%" if pct_dev.is_integer() else f"{pct_dev:.1f}%"
        fmt_teste = f"{pct_teste:.0f}%" if pct_teste.is_integer() else f"{pct_teste:.1f}%"
        logger.info(
            "[Passo 4/%d] Separando base em Desenvolvimento (%s) e Holdout de Teste (%s)...",
            self._config.total_passos,
            fmt_dev,
            fmt_teste,
        )
        df_dev, df_holdout = self.separar_holdout(dados)
        logger.info(" -> Base de Desenvolvimento: %d amostras | Holdout Final: %d amostras.", len(df_dev), len(df_holdout))

        X_dev, y_dev = self.extrair_features_alvo(df_dev)
        X_holdout, y_holdout = self.extrair_features_alvo(df_holdout)

        # 5. Criar candidatos
        tipos_candidatos = self.criar_candidatos()

        # 6. GridSearchCV
        pipelines_otimizadas, _ = self.executar_grid_search(tipos_candidatos, X_dev, y_dev)

        # 6.1 Extração de equações / importâncias por modelo com melhores parâmetros
        logger.info("[Passo 6.1/%d] Extraindo equação/importâncias de cada modelo candidato (melhores parâmetros + treino em 100%% do dev)...", self._config.total_passos)
        self.extrair_explicabilidade_candidatos(pipelines_otimizadas, X_dev, y_dev)

        # 7. RepeatedKFold (30 repetições) com mesmas partições
        res_cv = self.validar_repeated_kfold(pipelines_otimizadas, X_dev, y_dev)

        # 8. Testes estatísticos pós-validação cruzada: Friedman e Nemenyi
        res_friedman, res_nemenyi = self.executar_testes_estatisticos(res_cv)

        # 9. Formação do grupo estatisticamente elegível
        grupo = self.selecionar_grupo_elegivel(res_cv, res_friedman, res_nemenyi)
        nome_individual, pipeline_individual = self.selecionar_modelo_individual(pipelines_otimizadas, grupo)
        candidato_campeao_nome = nome_individual
        pipeline_campea = pipeline_individual

        # 10. Avaliação de votação / ensemble se usar_votacao=True
        if self._config.usar_votacao and len(grupo.modelos_elegiveis) >= 2:
            res_ensemble = self.avaliar_ensembles(pipelines_otimizadas, grupo, X_dev, y_dev)
            candidato_campeao_nome, pipeline_campea = self.comparar_candidatos(
                nome_individual=nome_individual,
                pipeline_individual=pipeline_individual,
                ensemble_resultado=res_ensemble,
                grupo=grupo,
            )
        else:
            logger.info(" -> Ensemble desativado ou modelos elegíveis insuficientes. Mantido modelo [%s].", candidato_campeao_nome)
            self.notificar(
                TipoEvento.ENSEMBLE_FINALIZADO,
                CargaEvento(valores={"usado": False}),
            )

        # 11. Treino final do campeão em todo o conjunto de desenvolvimento
        pipeline_campea = self.treinar_final(pipeline_campea, X_dev, y_dev, candidato_campeao_nome)

        # 12. Avaliação única no Holdout Final
        rmse_cv_campeao = (
            res_ensemble[2].rmse
            if "res_ensemble" in locals() and candidato_campeao_nome == res_ensemble[0]
            else grupo.rmse_medios.get(candidato_campeao_nome)
        )
        y_pred_holdout, metricas_holdout = self.avaliar_holdout(
            pipeline_campea=pipeline_campea,
            X_holdout=X_holdout,
            y_holdout=y_holdout,
            nome_campeao=candidato_campeao_nome,
            rmse_cv=rmse_cv_campeao,
        )

        # 13. Diagnóstico de Underfitting/Overfitting via Curva de Aprendizado
        self.gerar_curva_aprendizado(pipeline_campea, X_dev, y_dev)

        # 14. Explicabilidade
        resultado_explicabilidade = self.extrair_explicabilidade(pipeline_campea, X_holdout, y_holdout, candidato_campeao_nome)

        # 15. Métricas de Negócio Imobiliário no Holdout
        metricas_negocio = self.calcular_metricas_negocio(
            X_holdout, y_holdout, y_pred_holdout, candidato_campeao_nome
        )

        # 16. Registro do Modelo Campeão no Model Registry do MLflow
        uri_registro = self.registrar_modelo(
            pipeline_campea, metricas_holdout, candidato_campeao_nome, metricas_negocio
        )

        logger.info("================================================================================")
        logger.info("PIPELINE DE TREINAMENTO CONCLUÍDO COM SUCESSO!")
        logger.info("================================================================================")

        return ResultadoTreinamento(
            modelo=pipeline_campea,
            metricas=metricas_holdout,
            nome_campeao=candidato_campeao_nome,
            metricas_negocio=metricas_negocio,
            explicabilidade=resultado_explicabilidade,
            uri_registro=uri_registro,
        )
