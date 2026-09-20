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
from ..preprocessamento_pkg.fabrica_escalonadores import FabricaEscalonadores
from ..preprocessamento_pkg.pre_processador import PreProcessador
from ..preprocessamento_pkg.tratador_categorico import TratadorCategorico
from ..qualidade_pkg.validador_esquema import ValidadorEsquema
from ..qualidade_pkg.validador_qualidade import ValidadorQualidade
from ..selecao_pkg.seletor_hiperparametros import SeletorHiperparametros
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
        self._registrador = RegistradorModelo(nome_modelo=self._config.mlflow_nome_modelo)

        # Adiciona automaticamente o observador oficial do MLflow
        self.adicionar_observador(
            ObservadorMlflow(
                nome_experimento=self._config.mlflow_experimento,
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
        }
        self.notificar(TipoEvento.EDA_FINALIZADA, CargaEvento(valores=payload))

    def separar_holdout(
        self, dados: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return train_test_split(  # type: ignore[return-value]
            dados, test_size=0.20, random_state=self._config.seed
        )

    def _obter_modelos_candidatos(self) -> list[TipoModelo]:
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

    def executar(self, caminho_dados: str | Path) -> ResultadoTreinamento[RegressorProtocol]:
        """Template Method que executa o fluxo oficial completo do projeto.

        Parameters
        ----------
        caminho_dados : str | Path
            Caminho do arquivo de dados (CSV).

        Returns
        -------
        ResultadoTreinamento[RegressorProtocol]
            Modelo campeão treinado, avaliado e registrado.
        """
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

        # 1. Carregar dados
        logger.info("================================================================================")
        logger.info("INICIANDO PIPELINE OFICIAL DE MACHINE LEARNING — PREVISÃO DE PREÇO DE IMÓVEIS")
        logger.info("Variável Alvo: %s | Semente Aleatória: %d", self._config.alvo, self._config.seed)
        logger.info("================================================================================")
        logger.info("[Passo 1/12] Carregando base de dados a partir de: %s", caminho_dados)
        dados = self.carregar_dados(caminho_dados)
        logger.info(" -> Base carregada com sucesso: %d registros e %d colunas.", len(dados), len(dados.columns))

        # 2. Validar esquema e qualidade
        logger.info("[Passo 2/12] Validando conformidade de esquema e integridade de qualidade...")
        self.validar_dados(dados)
        logger.info(" -> Validação concluída: colunas obrigatórias presentes, sem duplicatas e dados válidos.")

        # 3. EDA obrigatória antes do treino
        logger.info("[Passo 3/12] Executando Análise Exploratória de Dados (EDA) obrigatória...")
        self.executar_eda(dados)
        logger.info(" -> EDA finalizada: correlações, estatísticas e gráficos de distribuição gerados em memória.")

        # 4. Separação desenvolvimento / holdout final
        logger.info("[Passo 4/12] Separando base em Desenvolvimento (80%%) e Holdout de Teste (20%%)...")
        df_dev, df_holdout = self.separar_holdout(dados)
        logger.info(" -> Base de Desenvolvimento: %d amostras | Holdout Final: %d amostras.", len(df_dev), len(df_holdout))

        X_dev = df_dev[list(self.COLUNAS_NUMERICAS) + list(self.COLUNAS_CATEGORICAS)]
        y_dev = df_dev[self._config.alvo]

        X_holdout = df_holdout[list(self.COLUNAS_NUMERICAS) + list(self.COLUNAS_CATEGORICAS)]
        y_holdout = df_holdout[self._config.alvo]

        # 5. Criar candidatos e 6. GridSearchCV
        tipos_candidatos = self._obter_modelos_candidatos()
        pipelines_otimizadas: dict[str, Pipeline] = {}
        candidatos_modelos: dict[str, CandidatoModelo] = {}

        logger.info("[Passo 5/12] Otimizando hiperparâmetros (GridSearchCV com %d splits) para %d modelos candidatos...", self._config.n_splits, len(tipos_candidatos))

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

            self.notificar(
                TipoEvento.GRIDSEARCH_FINALIZADO,
                CargaEvento(
                    valores={
                        "nome_modelo": tipo.value,
                        "melhores_parametros": res_grid.melhores_parametros,
                        "melhor_score": res_grid.melhor_score_rmse,
                        "tabela_cv_results": res_grid.tabela_cv_results,
                    }
                ),
            )

            # Congela os melhores hiperparâmetros na pipeline do modelo
            pipeline_otimizada = clone(pipeline_base)
            pipeline_otimizada.set_params(**res_grid.melhores_parametros)
            pipelines_otimizadas[tipo.value] = pipeline_otimizada

            candidatos_modelos[tipo.value] = CandidatoModelo(
                tipo_modelo=tipo,
                nome=tipo.value,
                pipeline=pipeline_otimizada,
                melhores_parametros=res_grid.melhores_parametros,
                rmse_cv=res_grid.melhor_score_rmse,
            )

        # 7. RepeatedKFold (30 repetições) com mesmas partições
        logger.info("[Passo 6/12] Executando validação cruzada RepeatedKFold (%d splits x %d repetições = %d dobras por modelo)...", self._config.n_splits, self._config.n_repeats, self._config.n_splits * self._config.n_repeats)
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

        # 8. Testes estatísticos pós-validação cruzada: Friedman e Nemenyi
        logger.info("[Passo 7/12] Aplicando testes estatísticos pós-validação cruzada (Friedman & Nemenyi)...")
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

        # 9. Formação do grupo estatisticamente elegível
        logger.info("[Passo 8/12] Delimitando o Grupo Estatisticamente Elegível...")
        grupo = self._seletor_campeao.selecionar_grupo(res_cv, res_friedman, res_nemenyi)
        melhor_individual_nome = grupo.melhor_modelo
        candidato_campeao_nome = melhor_individual_nome
        pipeline_campea = pipelines_otimizadas[melhor_individual_nome]
        logger.info(" -> Melhor modelo individual: [%s] (RMSE médio CV: R$ %s)", melhor_individual_nome, f"{grupo.rmse_medios.get(melhor_individual_nome, 0.0):,.2f}")
        logger.info(" -> Modelos elegíveis (%d): %s", len(grupo.modelos_elegiveis), ", ".join(grupo.modelos_elegiveis))

        # 10. Avaliação de votação / ensemble se usar_votacao=True
        logger.info("[Passo 9/12] Avaliando estratégias de Ensemble (usar_votacao = %s)...", self._config.usar_votacao)
        if self._config.usar_votacao and len(grupo.modelos_elegiveis) >= 2:
            modelos_elegiveis = self._seletor_ensemble.filtrar_elegiveis(
                pipelines_otimizadas, grupo
            )
            # Avalia votação ponderada por RMSE
            pesos_rmse = self._calculador_pesos_rmse.calcular(grupo.rmse_medios)
            logger.info(" -> Modelos que compõem o Ensemble: %s", [m[0] for m in modelos_elegiveis])
            logger.info(" -> Pesos atribuídos por RMSE: %s", {k: f"{v:.4f}" for k, v in pesos_rmse.items()})

            estrategia_voting = self._fabrica_ensemble.criar(
                TipoEnsemble.VOTING_PONDERADO_RMSE, pesos_por_modelo=pesos_rmse
            )
            ensemble_estimador = estrategia_voting.criar_estimador(
                [(nome, pipe.named_steps["modelo"]) for nome, pipe in modelos_elegiveis]
            )

            # Pré-processamento unificado para o ensemble
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

            # Valida ensemble nos mesmos dados de dev
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
                    }
                ),
            )

            # Compara com o melhor individual
            rmse_melhor_ind = grupo.rmse_medios.get(melhor_individual_nome, float("inf"))
            if metricas_ens_dev.rmse < rmse_melhor_ind:
                logger.info(" -> Ensemble SUPEROU o melhor individual! (RMSE Ensemble: R$ %s vs Individual: R$ %s)", f"{metricas_ens_dev.rmse:,.2f}", f"{rmse_melhor_ind:,.2f}")
                candidato_campeao_nome = "ensemble_voting"
                pipeline_campea = pipeline_ensemble
            else:
                logger.info(" -> Melhor individual [%s] superou o Ensemble (R$ %s vs R$ %s). Mantido individual.", melhor_individual_nome, f"{rmse_melhor_ind:,.2f}", f"{metricas_ens_dev.rmse:,.2f}")
        else:
            logger.info(" -> Ensemble desativado ou modelos elegíveis insuficientes. Mantido modelo [%s].", candidato_campeao_nome)
            self.notificar(
                TipoEvento.ENSEMBLE_FINALIZADO,
                CargaEvento(valores={"usado": False}),
            )

        # 11. Treino final do campeão em todo o conjunto de desenvolvimento
        logger.info("[Passo 10/12] Realizando treinamento final do campeão [%s] em 100%% da base de desenvolvimento (%d registros)...", candidato_campeao_nome, len(X_dev))
        pipeline_campea.fit(X_dev, y_dev)
        logger.info(" -> Ajuste final do modelo campeão concluído com sucesso.")

        # 12. Avaliação única no Holdout Final
        logger.info("[Passo 11/12] Avaliando campeão no Holdout Final inédito (%d registros) e calculando métricas imobiliárias...", len(X_holdout))
        y_pred_holdout = np.asarray(pipeline_campea.predict(X_holdout), dtype=float)
        metricas_holdout = self._calculador_metricas.calcular(y_holdout, y_pred_holdout)

        logger.info("--------------------------------------------------------------------------------")
        logger.info(" DESEMPENHO DO MODELO CAMPEÃO [%s] NO HOLDOUT FINAL:", candidato_campeao_nome)
        logger.info("   * RMSE : R$ %s", f"{metricas_holdout.rmse:,.2f}")
        logger.info("   * MAE  : R$ %s", f"{metricas_holdout.mae:,.2f}")
        logger.info("   * R²   : %.4f", metricas_holdout.r2)
        logger.info("   * MAPE : %.2f%%", metricas_holdout.mape)
        logger.info("--------------------------------------------------------------------------------")

        self.notificar(
            TipoEvento.MODELO_CAMPEAO,
            CargaEvento(
                valores={
                    "nome_campeao": candidato_campeao_nome,
                    "metricas_holdout": {
                        "rmse": metricas_holdout.rmse,
                        "mae": metricas_holdout.mae,
                        "mse": metricas_holdout.mse,
                        "r2": metricas_holdout.r2,
                        "medae": metricas_holdout.medae,
                    },
                }
            ),
        )

        # 13. Diagnóstico de Underfitting/Overfitting via Curva de Aprendizado
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

        # 14. Explicabilidade
        logger.info(" -> Extraindo explicabilidade e importância das variáveis do campeão...")
        resultado_explicabilidade: ResultadoExplicabilidade | None = None
        estimador_interno = pipeline_campea.named_steps.get("modelo")
        if hasattr(estimador_interno, "coef_"):
            df_coef, intercepto, equacao = self._extrator_coef.extrair(pipeline_campea)
            md_interp = self._interpretador.gerar_interpretacao_markdown(df_coef, intercepto)
            resultado_explicabilidade = ResultadoExplicabilidade(
                tipo_modelo=candidato_campeao_nome,
                intercepto=intercepto,
                tabela_coeficientes=df_coef,
                equacao_texto=equacao,
                interpretacao_texto=md_interp,
            )
            logger.info(" -> Equação matemática extraída: %s", equacao)
        else:
            df_imp = self._explicador_imp.calcular_importancias(
                pipeline_ajustada=pipeline_campea,
                X_val=X_holdout,
                y_val=y_holdout,
            )
            resultado_explicabilidade = ResultadoExplicabilidade(
                tipo_modelo=candidato_campeao_nome,
                tabela_importancias=df_imp,
            )
            logger.info(" -> Importâncias por permutação calculadas para %d features.", len(df_imp))

        # 15. Métricas de Negócio Imobiliário no Holdout
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

        self.notificar(
            TipoEvento.METRICAS_NEGOCIO_FINALIZADAS,
            CargaEvento(
                valores={
                    "metricas_consolidadas": {
                        "cobertura_5": metricas_negocio.cobertura_5,
                        "cobertura_10": metricas_negocio.cobertura_10,
                        "cobertura_15": metricas_negocio.cobertura_15,
                        "mae_reais": metricas_negocio.mae_reais,
                        "vies_medio": metricas_negocio.vies_medio,
                        "desconto_seguro": metricas_negocio.desconto_seguro_recomendado,
                        "risco_subprecificacao": metricas_negocio.risco_subprecificacao,
                        "risco_superprecificacao": metricas_negocio.risco_superprecificacao,
                    },
                    "tabela_erro_faixa": metricas_negocio.tabela_erro_por_faixa,
                    "tabela_erro_zona": metricas_negocio.tabela_erro_por_zona,
                }
            ),
        )

        # 16. Registro do Modelo Campeão no Model Registry do MLflow
        logger.info("[Passo 12/12] Registrando modelo campeão encapsulado em PyFunc no MLflow Model Registry...")
        uri_registro = self._registrador.registrar_modelo_campeao(
            pipeline_campea=pipeline_campea,
            metricas={
                "rmse": metricas_holdout.rmse,
                "mae": metricas_holdout.mae,
                "r2": metricas_holdout.r2,
            },
            desconto_maximo=self._config.desconto_maximo_com_aprovacao,
        )
        logger.info(" -> Modelo registrado no MLflow com alias @champion: %s", uri_registro)
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
