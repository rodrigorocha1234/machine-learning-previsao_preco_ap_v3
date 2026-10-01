import pandas as pd
from sklearn.pipeline import Pipeline

from app_build.ajuste_modelos.explicador_parametros import ExplicadorParametros
from app_build.ajuste_modelos.fabrica_estimadores import FabricaEstimadores
from app_build.ajuste_modelos.fabrica_tuning import FabricaTuning
from app_build.camada_dados.analise_exploratoria import AnaliseExploratoria
from app_build.camada_dados.carregador_excel import CarregadorExcel
from app_build.camada_dados.deteccao_deriva import DeteccaoDeriva
from app_build.camada_dados.repositorio_staging import RepositorioStaging
from app_build.camada_dados.validador_contrato import ValidadorContrato
from app_build.configuracao_sistema.armazem_configuracao import ArmazemConfiguracao
from app_build.estatistica_modelos.contrato_estatistica import (
    ResultadoShapiro,
)
from app_build.estatistica_modelos.extrator_residuos import ExtratorResiduos
from app_build.estatistica_modelos.teste_friedman import TesteFriedman
from app_build.estatistica_modelos.teste_nemenyi import TesteNemenyi
from app_build.estatistica_modelos.teste_shapiro import TesteShapiro
from app_build.estatistica_modelos.teste_tukey import TesteTukey
from app_build.isolamento_dados.cofre_holdout import CofreHoldout
from app_build.isolamento_dados.divisor_estratificado import DivisorEstratificado
from app_build.orquestracao_pipeline.contexto_execucao import ContextoExecucao
from app_build.orquestracao_pipeline.contrato_etapa import ContratoEtapa
from app_build.processamento_dados.construtor_pipeline import ConstrutorPipeline
from app_build.rastreamento_mlflow.contrato_observador import (
    EventoEstatisticaConcluida,
    EventoHoldoutAvaliado,
    EventoModeloSelecionado,
    EventoNestedCvConcluido,
    EventoRegrasNegocioConcluido,
    EventoTreinoFinalConcluido,
)
from app_build.regras_negocio.agregador_hierarquico import AgregadorHierarquico
from app_build.regras_negocio.contrato_negocio import ResultadoImobiliario
from app_build.regras_negocio.motor_imobiliario import MotorImobiliario
from app_build.selecao_modelos.seletor_campeao import SeletorCampeao
from app_build.validacao_cruzada.acumulador_metricas import AcumuladorMetricas
from app_build.validacao_cruzada.avaliador_aninhado import AvaliadorAninhado
from app_build.validacao_cruzada.contrato_validador import MetricasRegressao
from app_build.validacao_cruzada.particionador_externo import ParticionadorExterno
from app_build.validacao_cruzada.particionador_interno import ParticionadorInterno


class Etapa01CarregarConfiguracoes(ContratoEtapa):
    @property
    def nome_etapa(self) -> str:
        return "01_carregar_configuracoes"

    def executar(self, contexto: ContextoExecucao) -> None:
        armazem = ArmazemConfiguracao()
        armazem.inicializar()
        contexto.configuracao_geral = armazem.geral
        contexto.configuracao_modelos = armazem.modelos


class Etapa02ValidarConfiguracoes(ContratoEtapa):
    @property
    def nome_etapa(self) -> str:
        return "02_validar_configuracoes"

    def executar(self, contexto: ContextoExecucao) -> None:
        assert contexto.configuracao_geral is not None, "Configuracao nao carregada"
        assert len(contexto.configuracao_modelos) > 0, "Nenhum modelo configurado"


class Etapa03CarregarDados(ContratoEtapa):
    @property
    def nome_etapa(self) -> str:
        return "03_carregar_dados"

    def executar(self, contexto: ContextoExecucao) -> None:
        carregador = CarregadorExcel()
        dados_carregados = carregador.carregar(
            "dados/bairro_final_v3_engineered.xlsx"
        )
        colunas_indesejadas: tuple[str, ...] = (
            "Código",
            "Apartamento",
            "valor_m2",
            "media_valor_m2_bairro",
            "media_valor_m2_zona",
        )
        cols_dados = set(dados_carregados.columns)
        colunas_para_remover: tuple[str, ...] = tuple(
            filter(cols_dados.__contains__, colunas_indesejadas)
        )
        contexto.dados_brutos = dados_carregados.drop(
            columns=list(colunas_para_remover)
        )


class Etapa04ValidarDados(ContratoEtapa):
    @property
    def nome_etapa(self) -> str:
        return "04_validar_dados"

    def executar(self, contexto: ContextoExecucao) -> None:
        assert contexto.dados_brutos is not None, "Dados brutos ausentes"
        assert contexto.configuracao_geral is not None, "Configuracao ausente"
        validador = ValidadorContrato(contexto.configuracao_geral.dados.target)
        validador.validar(contexto.dados_brutos)

        # ── Qualidade dos dados → Prometheus ──────────────────────────────────
        if contexto.coletor is not None:
            import numpy as np
            df = contexto.dados_brutos
            target = contexto.configuracao_geral.dados.target
            colunas_num = df.select_dtypes(include="number").columns.tolist()

            # Missing por coluna
            missing = {
                col: float(df[col].isna().mean() * 100)
                for col in df.columns
            }

            # Outliers por coluna numérica (método IQR)
            outliers: dict[str, float] = {}
            for col in colunas_num:
                q1 = float(df[col].quantile(0.25))
                q3 = float(df[col].quantile(0.75))
                iqr = q3 - q1
                n_out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
                outliers[col] = round(n_out / len(df) * 100, 2)

            # Estatísticas do target
            alvo = df[target].dropna() if target in df.columns else df.iloc[:, -1]
            amostras_zona: dict[str, int] = {}
            amostras_bairro: dict[str, int] = {}
            if "Zona" in df.columns:
                amostras_zona = df["Zona"].value_counts().to_dict()
            if "Bairro" in df.columns:
                amostras_bairro = df["Bairro"].value_counts().to_dict()

            contexto.coletor.registrar_qualidade_dados(
                total_amostras=len(df),
                media_alvo=float(alvo.mean()),
                mediana_alvo=float(alvo.median()),
                std_alvo=float(alvo.std()),
                assimetria_alvo=float(alvo.skew()),
                missing_por_coluna=missing,
                outliers_por_coluna=outliers,
                amostras_por_zona=amostras_zona,
                amostras_por_bairro=amostras_bairro,
            )
            contexto.coletor.registrar_recursos_sistema()


class Etapa05Staging(ContratoEtapa):
    @property
    def nome_etapa(self) -> str:
        return "05_executar_staging"

    def executar(self, contexto: ContextoExecucao) -> None:
        assert contexto.dados_brutos is not None, "Dados brutos ausentes"
        repo = RepositorioStaging()
        repo.salvar_snapshot(contexto.dados_brutos, "apartamentos_staging")


class Etapa06SepararHoldout(ContratoEtapa):
    @property
    def nome_etapa(self) -> str:
        return "06_separar_holdout"

    def executar(self, contexto: ContextoExecucao) -> None:
        assert contexto.dados_brutos is not None, "Dados brutos ausentes"
        assert contexto.configuracao_geral is not None, "Configuracao ausente"
        divisor = DivisorEstratificado(
            proporcao_holdout=contexto.configuracao_geral.dados.proporcao_holdout,
            semente=contexto.configuracao_geral.dados.semente_holdout,
        )
        particao = divisor.dividir(contexto.dados_brutos)
        contexto.dados_desenvolvimento = particao.desenvolvimento
        contexto.cofre_holdout.bloquear(particao.holdout)


class Etapa07BloquearHoldout(ContratoEtapa):
    @property
    def nome_etapa(self) -> str:
        return "07_bloquear_holdout"

    def executar(self, contexto: ContextoExecucao) -> None:
        pass


class Etapa08Eda(ContratoEtapa):
    @property
    def nome_etapa(self) -> str:
        return "08_executar_eda"

    def executar(self, contexto: ContextoExecucao) -> None:
        assert contexto.dados_desenvolvimento is not None, (
            "Base de desenvolvimento ausente"
        )
        assert contexto.configuracao_geral is not None, "Configuracao ausente"
        eda = AnaliseExploratoria(contexto.configuracao_geral.dados.target)
        eda.executar(contexto.dados_desenvolvimento)


class Etapa09Drift(ContratoEtapa):
    @property
    def nome_etapa(self) -> str:
        return "09_executar_drift"

    def executar(self, contexto: ContextoExecucao) -> None:
        assert contexto.dados_desenvolvimento is not None, (
            "Base de desenvolvimento ausente"
        )
        features_drift = ("Metragem", "Quartos", "Banheiros", "Vagas_Garagem")

        # ── DeteccaoDeriva (KS por coluna) — comportamento original ─────────────
        detector_ks = DeteccaoDeriva()
        detector_ks.comparar(
            contexto.dados_desenvolvimento,
            contexto.dados_desenvolvimento,
            features_drift,
        )

        # ── DetectorDrift (PSI + KS + Wasserstein) → Prometheus ─────────────────
        if contexto.coletor is not None:
            from app_build.observabilidade_metricas.detector_drift import DetectorDrift
            import numpy as np

            detector = DetectorDrift()
            df = contexto.dados_desenvolvimento

            psi_total = 0.0
            ks_stat_area = 0.0
            ks_pval_area = 1.0
            wasserstein_area = 0.0
            status_geral = 0

            for feature in features_drift:
                if feature not in df.columns:
                    continue
                arr = df[feature].dropna().to_numpy(dtype=float)
                resultado = detector.analisar_distribuicao(arr, arr)

                contexto.coletor.registrar_drift_feature(
                    feature=feature, psi=resultado.psi
                )
                psi_total += resultado.psi

                # Usa Metragem como feature principal para métricas globais
                if feature == "Metragem":
                    ks_stat_area = resultado.ks_estatistica
                    ks_pval_area = resultado.ks_pvalor
                    wasserstein_area = resultado.wasserstein_distancia
                    status_geral = resultado.status_severidade

            contexto.coletor.registrar_drift_dados(
                psi_predicoes=psi_total / max(len(features_drift), 1),
                psi_area=psi_total / max(len(features_drift), 1),
                ks_stat_area=ks_stat_area,
                ks_pval_area=ks_pval_area,
                wasserstein_area=wasserstein_area,
                status_geral=status_geral,
            )

            # Desvio de preço por zona (se disponível)
            target_col = (
                contexto.configuracao_geral.dados.target
                if contexto.configuracao_geral is not None
                else "Valor"
            )
            if "Zona" in df.columns and target_col in df.columns:
                mediana_global = float(df[target_col].median()) + 1e-6
                for zona, grupo in df.groupby("Zona"):
                    mediana_zona = float(grupo[target_col].median())
                    desvio = (mediana_zona - mediana_global) / mediana_global * 100.0
                    contexto.coletor.registrar_desvio_preco_zona(
                        zona=str(zona), desvio_percentual=desvio
                    )


class Etapa10NestedCv(ContratoEtapa):
    @property
    def nome_etapa(self) -> str:
        return "10_executar_nested_cv"

    def executar(self, contexto: ContextoExecucao) -> None:
        assert contexto.dados_desenvolvimento is not None, (
            "Base de desenvolvimento ausente"
        )
        assert contexto.configuracao_geral is not None, "Configuracao ausente"

        target_col = contexto.configuracao_geral.dados.target
        dados_x = contexto.dados_desenvolvimento.drop(columns=[target_col])
        vetor_y = contexto.dados_desenvolvimento[target_col]

        particionador_ext = ParticionadorExterno(
            splits=contexto.configuracao_geral.validacao.splits_externos,
            repeticoes=contexto.configuracao_geral.validacao.repeticoes_externas,
            semente=contexto.configuracao_geral.validacao.semente_externa,
        )
        divisoes = particionador_ext.gerar_divisoes(dados_x)
        contexto.divisoes_externas = divisoes

        explicador = ExplicadorParametros()
        modelos_ativos = filter(
            lambda item: bool(item[1].get("ativo", True)),
            contexto.configuracao_modelos.items(),
        )

        for nome_modelo, cfg_modelo in modelos_ativos:
            avaliador = AvaliadorAninhado(
                configuracao_modelo=cfg_modelo,
                divisoes_externas=divisoes,
                particionador_interno=ParticionadorInterno(
                    splits=contexto.configuracao_geral.validacao.splits_internos,
                    embaralhar=contexto.configuracao_geral.validacao.embaralhar_interno,
                    semente=contexto.configuracao_geral.validacao.semente_interna,
                ),
                construtor_pipeline=ConstrutorPipeline(),
            )
            resultado_cv = avaliador.avaliar_modelo(nome_modelo, dados_x, vetor_y)
            contexto.resultados_nested_cv[nome_modelo] = resultado_cv

            params_explicados = explicador.explicar(
                nome_modelo,
                resultado_cv.resultados_folds[0].melhores_parametros,
            )
            contexto.despachante.despachar_nested_cv(
                EventoNestedCvConcluido(
                    nome_modelo=nome_modelo,
                    resultado_cv=resultado_cv,
                    explicacoes_parametros=params_explicados,
                )
            )

            # ── CV por fold → Prometheus ───────────────────────────────────────
            if contexto.coletor is not None:
                import numpy as np
                contexto.coletor.registrar_dispersao_cv(resultado_cv)
                for fold in resultado_cv.resultados_folds:
                    m = fold.metricas
                    contexto.coletor.registrar_resultado_fold_cv(
                        modelo=nome_modelo,
                        fold=fold.indice_fold,
                        rmse=m.rmse,
                        mae=m.mae,
                        r2=m.r2,
                        mape=m.mape,
                    )

                tempos = [f.tempo_segundos for f in resultado_cv.resultados_folds]
                contexto.coletor.registrar_resumo_cv_modelo(
                    modelo=nome_modelo,
                    rmse_medio=resultado_cv.metricas_medias.rmse,
                    r2_medio=resultado_cv.metricas_medias.r2,
                    rmse_std=resultado_cv.metricas_desvios_padrao.rmse,
                    duracao_media_fold_s=float(np.mean(tempos)),
                )
                contexto.coletor.registrar_recursos_sistema()


class Etapa11Estatistica(ContratoEtapa):
    @property
    def nome_etapa(self) -> str:
        return "11_executar_estatistica"

    def executar(self, contexto: ContextoExecucao) -> None:
        matriz_rmse = ExtratorResiduos.extrair_matriz_rmse(
            contexto.resultados_nested_cv
        )
        total_folds = len(matriz_rmse)

        teste_friedman = TesteFriedman()
        res_friedman = teste_friedman.testar(matriz_rmse)
        contexto.resultado_friedman = res_friedman

        teste_nemenyi = TesteNemenyi()
        res_nemenyi = teste_nemenyi.testar(res_friedman, total_folds)
        contexto.resultado_nemenyi = res_nemenyi

        teste_shapiro = TesteShapiro()
        resultados_shapiro: list[ResultadoShapiro] = []
        for nome_mod, res_cv in contexto.resultados_nested_cv.items():
            res_shapiro = teste_shapiro.testar(nome_mod, res_cv.residuos_totais)
            resultados_shapiro.append(res_shapiro)
        contexto.resultados_shapiro = resultados_shapiro

        teste_tukey = TesteTukey()
        teste_tukey.testar_anova(matriz_rmse)

        contexto.despachante.despachar_estatistica(
            EventoEstatisticaConcluida(
                resultado_friedman=res_friedman,
                resultado_nemenyi=res_nemenyi,
                resultados_shapiro=tuple(resultados_shapiro),
            )
        )


class Etapa12SelecaoModelo(ContratoEtapa):
    @property
    def nome_etapa(self) -> str:
        return "12_selecionar_modelo"

    def executar(self, contexto: ContextoExecucao) -> None:
        assert contexto.configuracao_geral is not None, "Configuracao ausente"
        assert contexto.resultado_friedman is not None, "Friedman ausente"

        seletor = SeletorCampeao(
            usar_votacao=contexto.configuracao_geral.selecao.usar_votacao,
            quantidade_modelos=contexto.configuracao_geral.selecao.quantidade_modelos,
        )
        decisao = seletor.executar_selecao(
            contexto.resultados_nested_cv,
            contexto.resultado_friedman,
        )
        contexto.decisao_selecao = decisao
        contexto.despachante.despachar_selecao(EventoModeloSelecionado(decisao))


class Etapa13TuningFinal(ContratoEtapa):
    @property
    def nome_etapa(self) -> str:
        return "13_tuning_final"

    def executar(self, contexto: ContextoExecucao) -> None:
        assert contexto.dados_desenvolvimento is not None, (
            "Base de desenvolvimento ausente"
        )
        assert contexto.configuracao_geral is not None, "Configuracao ausente"
        assert contexto.decisao_selecao is not None, "Decisao de selecao ausente"

        target_col = contexto.configuracao_geral.dados.target
        dados_x = contexto.dados_desenvolvimento.drop(columns=[target_col])
        vetor_y = contexto.dados_desenvolvimento[target_col]

        nome_campeao = contexto.decisao_selecao.modelo_principal
        cfg_modelo = contexto.configuracao_modelos[nome_campeao]

        from collections.abc import Mapping
        from typing import cast

        parametros_base = dict(
            cast(Mapping[str, object], cfg_modelo.get("parametros", {}))
        )
        secao_tuning = dict(cast(Mapping[str, object], cfg_modelo.get("tuning", {})))
        nome_estrategia = str(secao_tuning.get("estrategia", "nenhum"))
        espaco_params = dict(
            cast(Mapping[str, object], secao_tuning.get("parametros", {}))
        )
        metrica_scoring = str(
            secao_tuning.get("scoring", "neg_root_mean_squared_error")
        )
        n_iter: int = cast(int, secao_tuning.get("n_iter", 10))
        semente: int = cast(int, secao_tuning.get("random_state", 42))

        estrategia_tuning = FabricaTuning.obter_estrategia(nome_estrategia)
        validador_interno = ParticionadorInterno(
            splits=contexto.configuracao_geral.validacao.splits_internos,
            embaralhar=contexto.configuracao_geral.validacao.embaralhar_interno,
            semente=contexto.configuracao_geral.validacao.semente_interna,
        ).obter_validador()

        estimador_base = FabricaEstimadores.criar_estimador(
            nome_campeao, parametros_base
        )
        preprocessador = ConstrutorPipeline().criar_preprocessador()
        pipeline_base = Pipeline(
            [
                ("preprocessamento", preprocessador),
                ("modelo", estimador_base),
            ]
        )

        resultado_tuning = estrategia_tuning.executar_tuning(
            estimador_base=pipeline_base,
            espaco_parametros=espaco_params,
            dados_x=dados_x,
            vetor_y=vetor_y,
            validador_cv=validador_interno,
            metrica_scoring=metrica_scoring,
            semente=semente,
            iteracoes=n_iter,
        )
        contexto.modelo_campeao_final = resultado_tuning.melhor_estimador


class Etapa14TreinamentoFinal(ContratoEtapa):
    @property
    def nome_etapa(self) -> str:
        return "14_treinamento_final"

    def executar(self, contexto: ContextoExecucao) -> None:
        assert contexto.dados_desenvolvimento is not None, (
            "Base de desenvolvimento ausente"
        )
        assert contexto.configuracao_geral is not None, "Configuracao ausente"
        assert contexto.modelo_campeao_final is not None, "Modelo final ausente"

        target_col = contexto.configuracao_geral.dados.target
        dados_x = contexto.dados_desenvolvimento.drop(columns=[target_col])
        vetor_y = contexto.dados_desenvolvimento[target_col]

        contexto.modelo_campeao_final.fit(dados_x, vetor_y)  # type: ignore[union-attr]

        agregador = AgregadorHierarquico(
            minima_zona=contexto.configuracao_geral.amostra.minima_zona,
            minima_bairro=contexto.configuracao_geral.amostra.minima_bairro,
        )
        estatisticas = agregador.calcular_estatisticas(
            contexto.dados_desenvolvimento, target_col
        )
        contexto.motor_imobiliario = MotorImobiliario(estatisticas)


class Etapa15CongelarConfiguracao(ContratoEtapa):
    @property
    def nome_etapa(self) -> str:
        return "15_congelar_configuracao"

    def executar(self, contexto: ContextoExecucao) -> None:
        assert contexto.modelo_campeao_final is not None, "Modelo nao ajustado"
        assert contexto.motor_imobiliario is not None, "Regras imobiliarias ausentes"
        contexto.estado_congelado = True


class Etapa16AbrirHoldout(ContratoEtapa):
    @property
    def nome_etapa(self) -> str:
        return "16_abrir_holdout"

    def executar(self, contexto: ContextoExecucao) -> None:
        assert contexto.estado_congelado, (
            "Violacao: Impossivel abrir holdout antes de congelar configuracao final"
        )


class Etapa17AvaliacaoHoldout(ContratoEtapa):
    @property
    def nome_etapa(self) -> str:
        return "17_avaliar_holdout"

    def executar(self, contexto: ContextoExecucao) -> None:
        assert contexto.configuracao_geral is not None, "Configuracao ausente"
        assert contexto.modelo_campeao_final is not None, "Modelo final ausente"

        target_col = contexto.configuracao_geral.dados.target
        dados_holdout = contexto.cofre_holdout.liberar_holdout(
            CofreHoldout.CHAVE_MESTRA
        )
        contexto.dados_holdout_liberados = dados_holdout

        dados_x = dados_holdout.drop(columns=[target_col])
        vetor_y = dados_holdout[target_col]

        previsoes = contexto.modelo_campeao_final.predict(dados_x)  # type: ignore[union-attr]
        metricas_global = AcumuladorMetricas.calcular_metricas(
            vetor_y.to_numpy(), previsoes
        )
        contexto.metricas_holdout_global = metricas_global

        df_com_prev = dados_holdout.assign(previsao=previsoes)

        metricas_zona: dict[str, MetricasRegressao] = {}
        grupos_zona_validos = filter(
            lambda item: len(item[1]) > 1, df_com_prev.groupby("Zona")
        )
        for zona, grupo in grupos_zona_validos:
            metricas_zona[str(zona)] = AcumuladorMetricas.calcular_metricas(
                grupo[target_col].to_numpy(),
                grupo["previsao"].to_numpy(),
            )
        contexto.metricas_holdout_zona = metricas_zona

        metricas_bairro: dict[str, MetricasRegressao] = {}
        grupos_bairro_validos = filter(
            lambda item: len(item[1]) > 1, df_com_prev.groupby("Bairro")
        )
        for bairro, grupo in grupos_bairro_validos:
            metricas_bairro[str(bairro)] = AcumuladorMetricas.calcular_metricas(
                grupo[target_col].to_numpy(),
                grupo["previsao"].to_numpy(),
            )
        contexto.metricas_holdout_bairro = metricas_bairro

        contexto.despachante.despachar_holdout(
            EventoHoldoutAvaliado(
                metricas_global=metricas_global,
                metricas_zona=metricas_zona,
                metricas_bairro=metricas_bairro,
            )
        )

        # ── Holdout granular + distribuição predições → Prometheus ────────────
        if contexto.coletor is not None:
            import numpy as np

            # Zona
            bairro_zona_map: dict[str, str] = {}
            if "Bairro" in dados_holdout.columns and "Zona" in dados_holdout.columns:
                bairro_zona_map = (
                    dados_holdout[["Bairro", "Zona"]]
                    .drop_duplicates()
                    .set_index("Bairro")["Zona"]
                    .to_dict()
                )
            for zona, m in metricas_zona.items():
                qtd = int((df_com_prev["Zona"] == zona).sum())
                contexto.coletor.registrar_holdout_zona(
                    zona=zona, rmse=m.rmse, mae=m.mae,
                    r2=m.r2, mape=m.mape, total_amostras=qtd,
                    valor_medio_previsto=float(
                        df_com_prev.loc[df_com_prev["Zona"] == zona, "previsao"].mean()
                    ),
                )
            for bairro, m in metricas_bairro.items():
                zona_do_bairro = bairro_zona_map.get(bairro, "Desconhecida")
                contexto.coletor.registrar_holdout_bairro(
                    bairro=bairro, zona=zona_do_bairro,
                    rmse=m.rmse, r2=m.r2,
                    valor_medio_previsto=float(
                        df_com_prev.loc[df_com_prev["Bairro"] == bairro, "previsao"].mean()
                    ),
                )

            # Distribuição dos resíduos e percentis do erro percentual
            y_real = vetor_y.to_numpy(dtype=float)
            res = y_real - previsoes
            erros_pct = np.abs(res / np.maximum(y_real, 1.0)) * 100.0
            contexto.coletor.registrar_distribuicao_predicoes(
                valores_previstos=previsoes.tolist(),
                residuos=res.tolist(),
                erros_percentuais_abs=erros_pct.tolist(),
            )
            contexto.coletor.registrar_recursos_sistema()


class Etapa18RegrasNegocio(ContratoEtapa):
    @property
    def nome_etapa(self) -> str:
        return "18_executar_regras_negocio"

    def executar(self, contexto: ContextoExecucao) -> None:
        assert contexto.motor_imobiliario is not None, "Motor imobiliario ausente"
        assert contexto.modelo_campeao_final is not None, "Modelo campeao ausente"
        assert contexto.configuracao_geral is not None, "Configuracao ausente"
        assert contexto.dados_holdout_liberados is not None, (
            "Dados do holdout liberados ausentes"
        )

        target_col = contexto.configuracao_geral.dados.target
        dados_holdout = contexto.dados_holdout_liberados
        dados_x = dados_holdout.drop(columns=[target_col])
        previsoes = contexto.modelo_campeao_final.predict(dados_x)  # type: ignore[union-attr]

        vetor_prev = pd.Series(previsoes, index=dados_x.index)
        df_enriquecido = contexto.motor_imobiliario.enriquecer_dataframe(
            dados_x, vetor_prev
        )
        df_enriquecido[target_col] = dados_holdout[target_col]

        exemplos: list[ResultadoImobiliario] = [
            contexto.motor_imobiliario.enriquecer_previsao(
                valor_previsto=850000.0,
                metragem=120.0,
                zona="Zona Sul",
                bairro="Jardim Botânico",
            ),
            contexto.motor_imobiliario.enriquecer_previsao(
                valor_previsto=1350000.0,
                metragem=180.0,
                zona="Zona Sul",
                bairro="Jardim Olhos D'Água",
            ),
            contexto.motor_imobiliario.enriquecer_previsao(
                valor_previsto=420000.0,
                metragem=65.0,
                zona="Zona Leste",
                bairro="Ribeirânia",
            ),
        ]

        metricas_negocio: dict[str, float] = {
            "indice_medio_imovel_global": float(
                df_enriquecido["indice_imovel_global"].mean()
            ),
            "indice_medio_imovel_zona": float(
                df_enriquecido["indice_imovel_zona"].mean()
            ),
            "indice_medio_imovel_bairro": float(
                df_enriquecido["indice_imovel_bairro"].mean()
            ),
            "dif_perc_media_global": float(
                df_enriquecido["diferenca_perc_global"].mean()
            ),
            "dif_perc_media_zona": float(
                df_enriquecido["diferenca_perc_zona"].mean()
            ),
            "dif_perc_media_bairro": float(
                df_enriquecido["diferenca_perc_bairro"].mean()
            ),
            "desconto_5_medio": float(df_enriquecido["bairro_desconto_5"].mean()),
            "desconto_10_medio": float(df_enriquecido["bairro_desconto_10"].mean()),
            "desconto_15_medio": float(df_enriquecido["bairro_desconto_15"].mean()),
            "faixa_segura_piso_media": float(
                df_enriquecido["bairro_faixa_segura_piso"].mean()
            ),
            "faixa_segura_teto_media": float(
                df_enriquecido["bairro_faixa_segura_teto"].mean()
            ),
        }

        contexto.despachante.despachar_regras_negocio(
            EventoRegrasNegocioConcluido(
                estatisticas_hierarquicas=contexto.motor_imobiliario.estatisticas,
                dados_enriquecidos=df_enriquecido,
                exemplos_simulacao=tuple(exemplos),
                metricas_negocio=metricas_negocio,
            )
        )


class Etapa19RastreamentoMlflow(ContratoEtapa):
    @property
    def nome_etapa(self) -> str:
        return "19_rastrear_mlflow"

    def executar(self, contexto: ContextoExecucao) -> None:
        assert contexto.decisao_selecao is not None, "Decisao de selecao ausente"
        assert contexto.modelo_campeao_final is not None, "Modelo campeao ausente"
        assert contexto.dados_desenvolvimento is not None, (
            "Base de desenvolvimento ausente"
        )
        assert contexto.configuracao_geral is not None, "Configuracao ausente"

        target_col = contexto.configuracao_geral.dados.target
        colunas_descartar: tuple[str, ...] = (
            target_col,
            "Código",
            "Apartamento",
            "valor_m2",
            "media_valor_m2_bairro",
            "media_valor_m2_zona",
        )
        cols_desenvolvimento = set(contexto.dados_desenvolvimento.columns)
        colunas_para_remover: tuple[str, ...] = tuple(
            filter(cols_desenvolvimento.__contains__, colunas_descartar)
        )
        dados_x = contexto.dados_desenvolvimento.drop(
            columns=list(colunas_para_remover)
        )
        explicador = ExplicadorParametros()
        params_exp = explicador.explicar(contexto.decisao_selecao.modelo_principal, {})

        contexto.despachante.despachar_treino_final(
            EventoTreinoFinalConcluido(
                nome_modelo=contexto.decisao_selecao.modelo_principal,
                estimador=contexto.modelo_campeao_final,
                explicacoes_parametros=params_exp,
                dados_exemplo=dados_x,
                motor_imobiliario=contexto.motor_imobiliario,
            )
        )



class Etapa20Serving(ContratoEtapa):
    @property
    def nome_etapa(self) -> str:
        return "20_disponibilizar_serving"

    def executar(self, contexto: ContextoExecucao) -> None:
        pass
