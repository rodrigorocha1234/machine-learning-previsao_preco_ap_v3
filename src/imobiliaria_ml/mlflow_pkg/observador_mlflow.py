"""Observador concreto para envio de métricas, figuras e artefatos ao MLflow."""

import os
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from ..enums_pkg.tipo_evento import TipoEvento
from .evento_mlflow_protocol import EventoMlflowProtocol
from .observador import Observador



class ObservadorMlflow(Observador):
    """Implementação do Observador que converte eventos do pipeline em registros no MLflow."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        nome_experimento: str = "previsao-preco-imoveis",
        tags_padrao: dict[str, str] | None = None,
        usar_votacao: bool = False,
    ) -> None:
        self._tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self._nome_experimento = nome_experimento
        self._tags_padrao = tags_padrao or {}
        self._usar_votacao = usar_votacao
        self._child_run_ativa = False
        self._runs_modelos: dict[str, str] = {}
        self._nome_campeao: str | None = None
        self._configurar_mlflow()

    def _formatar_nome_run(self, nome_base: str) -> str:
        """Aplica o sufixo _votacao ao nome da run quando usar_votacao for True."""
        if self._usar_votacao:
            return f"{nome_base}_votacao"
        return nome_base

    def _configurar_mlflow(self) -> None:
        """Define o Tracking URI e o experimento ativo."""
        try:
            mlflow.set_tracking_uri(self._tracking_uri)
            client = MlflowClient(tracking_uri=self._tracking_uri)
            exp = client.get_experiment_by_name(self._nome_experimento)
            if exp is not None and exp.lifecycle_stage == "deleted":
                client.restore_experiment(exp.experiment_id)
            mlflow.set_experiment(self._nome_experimento)
        except Exception:
            # Permite execução isolada se o servidor estiver offline
            pass

    @staticmethod
    def _obter_familia_modelo(nome_modelo: str) -> str:
        """Determina a família algorítmica do modelo para categorização em tags."""
        nome_baixo = nome_modelo.lower()
        if any(term in nome_baixo for term in ("linear", "multipla", "polinomial", "ridge", "lasso", "elastic_net")):
            return "linear"
        if any(term in nome_baixo for term in ("arvore", "floresta", "random_forest")):
            return "arvore"
        if any(term in nome_baixo for term in ("boosting", "xgboost", "lightgbm", "catboost")):
            return "boosting"
        if "svr" in nome_baixo:
            return "kernel"
        if "neural" in nome_baixo or "mlp" in nome_baixo:
            return "rede_neural"
        if any(term in nome_baixo for term in ("ensemble", "voting", "stacking", "bagging")):
            return "ensemble"
        return "outros"

    @staticmethod
    def _gerar_figura_coberturas(c5: float, c10: float, c15: float) -> Figure:
        """Gera gráfico de barras horizontal das coberturas comerciais de erro do modelo campeão."""
        fig, ax = plt.subplots(figsize=(7, 3.5))
        faixas = ["±5% (Alta Precisão)", "±10% (Padrão Mercado)", "±15% (Margem Segurança)"]
        valores = [c5 * 100.0, c10 * 100.0, c15 * 100.0]
        cores = ["#2b5c8f", "#27ae60", "#16a085"]
        barras = ax.barh(faixas, valores, color=cores, height=0.5)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Imóveis Dentro da Tolerância (%)")
        ax.set_title("Cobertura Comercial de Erro - Modelo Campeão", fontsize=11, fontweight="bold")
        for barra in barras:
            largura = barra.get_width()
            ax.text(largura + 1.5, barra.get_y() + barra.get_height() / 2, f"{largura:.1f}%", va="center", fontsize=9)
        plt.tight_layout()
        return fig

    def _garantir_run_pai(self, nome_padrao: str = "pipeline_treinamento") -> None:
        """Garante que a run pai do pipeline esteja ativa com tags de governança."""
        if mlflow.active_run() is None:
            nome_run = self._formatar_nome_run(nome_padrao)
            mlflow.start_run(run_name=nome_run)
            tags_iniciais: dict[str, str] = {
                "mlflow.runName": nome_run,
                "projeto": "previsao_preco_imoveis",
                "tipo_tarefa": "regressao",
                "alvo": "Valor_da_Venda",
                "tipo_run": "pipeline",
                "framework": "scikit-learn",
                "ambiente": os.getenv("AMBIENTE", "desenvolvimento"),
                "usar_votacao": str(self._usar_votacao).lower(),
            }
            if self._tags_padrao:
                tags_iniciais.update(self._tags_padrao)
            mlflow.set_tags(tags_iniciais)

    def fechar_runs(self) -> None:
        """Encerra com segurança qualquer run que ainda esteja ativa no MLflow."""
        try:
            while mlflow.active_run() is not None:
                mlflow.end_run()
            self._child_run_ativa = False
            self._runs_modelos.clear()
        except Exception:
            pass

    def atualizar(
        self,
        evento: TipoEvento,
        dados: EventoMlflowProtocol,
    ) -> None:
        """Processa eventos e despacha diretamente para as APIs do MLflow sem arquivos locais permanentes.

        Parameters
        ----------
        evento : TipoEvento
            Identificador do evento emitido.
        dados : EventoMlflowProtocol
            Carga de dados associada ao evento.
        """
        payload = dados.para_dicionario()

        try:
            match evento:
                case TipoEvento.EDA_FINALIZADA:
                    self._garantir_run_pai()
                    mlflow.set_tag("etapa_eda", "concluida")
                    for k, v in payload.items():
                        if isinstance(v, pd.DataFrame):
                            mlflow.log_table(data=v, artifact_file=f"eda/{k}.json")
                        elif isinstance(v, Figure):
                            mlflow.log_figure(figure=v, artifact_file=f"eda/{k}.png")
                        elif isinstance(v, dict):
                            mlflow.log_dict(dictionary=v, artifact_file=f"eda/{k}.json")

                case TipoEvento.GRIDSEARCH_INICIADO:
                    nome = str(payload.get("nome_modelo", "modelo"))
                    self._garantir_run_pai()
                    if self._child_run_ativa:
                        try:
                            mlflow.end_run()
                        except Exception:
                            pass
                        self._child_run_ativa = False

                    # Inicia run filha nomeada com o nome do modelo e tags categorizadas
                    nome_run = self._formatar_nome_run(nome)
                    run_filha = mlflow.start_run(run_name=nome_run, nested=True)
                    self._child_run_ativa = True
                    self._runs_modelos[nome] = run_filha.info.run_id
                    self._runs_modelos[nome_run] = run_filha.info.run_id

                    familia = self._obter_familia_modelo(nome)
                    tags_candidato: dict[str, str] = {
                        "mlflow.runName": nome_run,
                        "modelo": nome,
                        "algoritmo": nome,
                        "familia_modelo": familia,
                        "tipo_run": "candidato",
                        "tipo_tarefa": "regressao",
                        "otimizador": "grid_search",
                        "projeto": "previsao_preco_imoveis",
                        "alvo": "Valor_da_Venda",
                        "usar_votacao": str(self._usar_votacao).lower(),
                    }
                    mlflow.set_tags(tags_candidato)
                    mlflow.log_param("nome_modelo", nome)

                case TipoEvento.GRIDSEARCH_FINALIZADO:
                    nome = str(payload.get("nome_modelo", "modelo"))
                    if "melhores_parametros" in payload and isinstance(payload["melhores_parametros"], dict):
                        mlflow.log_dict(
                            dictionary=payload["melhores_parametros"],
                            artifact_file=f"gridsearch/{nome}_melhores_parametros.json",
                        )
                        for pk, pv in payload["melhores_parametros"].items():
                            try:
                                mlflow.log_param(str(pk), str(pv))
                            except Exception:
                                pass

                    # Registra as métricas na mesma linha do modelo (run filha ativa)
                    if "melhor_score" in payload and isinstance(payload["melhor_score"], (int, float)):
                        score = float(payload["melhor_score"])
                        mlflow.log_metric("rmse_cv", score)
                        mlflow.log_metric("rmse", score)
                        mlflow.log_metric("melhor_score_cv", score)

                    if "tabela_cv_results" in payload and isinstance(payload["tabela_cv_results"], pd.DataFrame):
                        mlflow.log_table(
                            data=payload["tabela_cv_results"],
                            artifact_file=f"gridsearch/{nome}_cv_results.json",
                        )

                    # Adiciona tags de conclusão do modelo candidato
                    tags_conclusao: dict[str, str] = {
                        "status_gridsearch": "concluido",
                    }
                    if "melhor_score" in payload and isinstance(payload["melhor_score"], (int, float)):
                        tags_conclusao["melhor_rmse_cv"] = f"{float(payload['melhor_score']):,.2f}"
                    mlflow.set_tags(tags_conclusao)

                    # Encerra a run filha do modelo e retorna para a run pai
                    if self._child_run_ativa:
                        mlflow.end_run()
                        self._child_run_ativa = False

                case TipoEvento.VALIDACAO_FINALIZADA:
                    self._garantir_run_pai()
                    mlflow.set_tag("etapa_validacao_cv", "repeated_kfold_concluida")
                    if "tabela_folds" in payload and isinstance(payload["tabela_folds"], pd.DataFrame):
                        mlflow.log_table(
                            data=payload["tabela_folds"],
                            artifact_file="validacao/folds_repeated_kfold.json",
                        )
                    if "matriz_repeticoes" in payload and isinstance(payload["matriz_repeticoes"], pd.DataFrame):
                        mlflow.log_table(
                            data=payload["matriz_repeticoes"],
                            artifact_file="validacao/matriz_repeticoes.json",
                        )
                    if "resumo_modelos" in payload and isinstance(payload["resumo_modelos"], pd.DataFrame):
                        mlflow.log_table(
                            data=payload["resumo_modelos"],
                            artifact_file="validacao/resumo_modelos.json",
                        )
                        # Atualiza as métricas robustas do RepeatedKFold diretamente na linha de cada modelo candidato
                        client = MlflowClient(tracking_uri=self._tracking_uri)
                        for _, linha in payload["resumo_modelos"].iterrows():
                            mod_nome = str(linha.get("modelo", ""))
                            run_id_filha = self._runs_modelos.get(mod_nome)
                            if run_id_filha:
                                try:
                                    if "rmse_medio" in linha and pd.notna(linha["rmse_medio"]):
                                        client.log_metric(run_id_filha, "rmse_repeated_kfold", float(linha["rmse_medio"]))
                                    if "rmse_desvio" in linha and pd.notna(linha["rmse_desvio"]):
                                        client.log_metric(run_id_filha, "rmse_desvio", float(linha["rmse_desvio"]))
                                    if "mae_medio" in linha and pd.notna(linha["mae_medio"]):
                                        client.log_metric(run_id_filha, "mae", float(linha["mae_medio"]))
                                        client.log_metric(run_id_filha, "mae_repeated_kfold", float(linha["mae_medio"]))
                                    if "r2_medio" in linha and pd.notna(linha["r2_medio"]):
                                        client.log_metric(run_id_filha, "r2", float(linha["r2_medio"]))
                                        client.log_metric(run_id_filha, "r2_repeated_kfold", float(linha["r2_medio"]))
                                    if "tempo_treino_medio" in linha and pd.notna(linha["tempo_treino_medio"]):
                                        client.log_metric(run_id_filha, "tempo_treino_s", float(linha["tempo_treino_medio"]))
                                except Exception:
                                    pass

                case TipoEvento.FRIEDMAN_FINALIZADO:
                    self._garantir_run_pai()
                    sig_str = "significativo" if payload.get("significativo") else "nao_significativo"
                    mlflow.set_tag("teste_friedman", sig_str)
                    if "friedman" in payload and isinstance(payload["friedman"], dict):
                        mlflow.log_dict(payload["friedman"], "estatistica/friedman.json")
                    if "p_valor" in payload and isinstance(payload["p_valor"], (int, float)):
                        mlflow.log_metric("friedman_p_valor", float(payload["p_valor"]))
                    if "estatistica" in payload and isinstance(payload["estatistica"], (int, float)):
                        mlflow.log_metric("friedman_estatistica", float(payload["estatistica"]))

                case TipoEvento.NEMENYI_FINALIZADO:
                    self._garantir_run_pai()
                    mlflow.set_tag("teste_nemenyi", "executado")
                    if "matriz_p_valores" in payload and isinstance(payload["matriz_p_valores"], pd.DataFrame):
                        mlflow.log_table(
                            data=payload["matriz_p_valores"],
                            artifact_file="estatistica/nemenyi_pvalores.json",
                        )

                case TipoEvento.ENSEMBLE_FINALIZADO:
                    self._garantir_run_pai()
                    usado = bool(payload.get("usado", False))
                    estrategia = str(payload.get("estrategia", "voting"))
                    nome_ens = f"ensemble_{estrategia}"
                    nome_run_ens = self._formatar_nome_run(nome_ens)

                    if usado:
                        # Inicia child run nomeada para o ensemble com tags
                        run_ens = mlflow.start_run(run_name=nome_run_ens, nested=True)
                        self._runs_modelos[nome_ens] = run_ens.info.run_id
                        self._runs_modelos[nome_run_ens] = run_ens.info.run_id
                        tags_ensemble: dict[str, str] = {
                            "mlflow.runName": nome_run_ens,
                            "modelo": nome_ens,
                            "tipo_run": "ensemble",
                            "estrategia_ensemble": estrategia,
                            "familia_modelo": "ensemble",
                            "tipo_tarefa": "regressao",
                            "projeto": "previsao_preco_imoveis",
                            "alvo": "Valor_da_Venda",
                            "usar_votacao": str(self._usar_votacao).lower(),
                        }
                        mlflow.set_tags(tags_ensemble)
                        mlflow.log_param("nome_modelo", nome_ens)
                        mlflow.log_param("ensemble/estrategia", estrategia)
                        if "pesos" in payload and isinstance(payload["pesos"], dict):
                            mlflow.log_dict(payload["pesos"], "ensemble/pesos.json")
                            for mk, mv in payload["pesos"].items():
                                if isinstance(mv, (int, float)):
                                    mlflow.log_metric(f"peso_{mk}", float(mv))
                        # Registra métricas de desempenho na linha do ensemble
                        if "metricas" in payload and isinstance(payload["metricas"], dict):
                            for met_k, met_v in payload["metricas"].items():
                                if isinstance(met_v, (int, float)):
                                    mlflow.log_metric(met_k, float(met_v))
                            if "rmse" in payload["metricas"] and isinstance(payload["metricas"]["rmse"], (int, float)):
                                mlflow.log_metric("rmse_cv", float(payload["metricas"]["rmse"]))
                        mlflow.end_run()

                    mlflow.set_tag("ensemble_usado", str(usado).lower())
                    mlflow.log_param("ensemble/usado", usado)
                    if "estrategia" in payload:
                        mlflow.log_param("ensemble/estrategia", estrategia)
                    if "pesos" in payload and isinstance(payload["pesos"], dict):
                        mlflow.log_dict(payload["pesos"], "ensemble/pesos.json")

                case TipoEvento.MODELO_CAMPEAO:
                    self._garantir_run_pai()
                    nome_campeao = str(payload.get("nome_campeao", "campeao"))
                    self._nome_campeao = nome_campeao
                    nome_run_campeao = self._formatar_nome_run(f"campeao_{nome_campeao}")
                    # Nomeia a run principal com o nome do modelo campeão e tags de governança
                    tags_campeao: dict[str, str] = {
                        "mlflow.runName": nome_run_campeao,
                        "modelo": nome_campeao,
                        "modelo_campeao": nome_campeao,
                        "campeao_tipo": "ensemble" if "ensemble" in nome_campeao else "individual",
                        "status_pipeline": "campeao_selecionado",
                        "usar_votacao": str(self._usar_votacao).lower(),
                    }
                    mlflow.set_tags(tags_campeao)
                    mlflow.log_param("campeao/nome", nome_campeao)
                    mlflow.log_param("nome_modelo_campeao", nome_campeao)
                    if "metricas_holdout" in payload and isinstance(payload["metricas_holdout"], dict):
                        for k, v in payload["metricas_holdout"].items():
                            if isinstance(v, (int, float)):
                                mlflow.log_metric(f"holdout_{k}", float(v))
                        if "rmse" in payload["metricas_holdout"] and isinstance(payload["metricas_holdout"]["rmse"], (int, float)):
                            mlflow.log_metric("rmse", float(payload["metricas_holdout"]["rmse"]))
                        if "mae" in payload["metricas_holdout"] and isinstance(payload["metricas_holdout"]["mae"], (int, float)):
                            mlflow.log_metric("mae", float(payload["metricas_holdout"]["mae"]))
                        if "r2" in payload["metricas_holdout"] and isinstance(payload["metricas_holdout"]["r2"], (int, float)):
                            mlflow.log_metric("r2", float(payload["metricas_holdout"]["r2"]))
                    if "rmse_cv" in payload and isinstance(payload["rmse_cv"], (int, float)):
                        mlflow.log_metric("rmse_cv", float(payload["rmse_cv"]))

                    # Identifica a run filha do modelo campeão e registra tags e métricas de holdout nela também
                    run_id_campeao = self._runs_modelos.get(nome_campeao)
                    if run_id_campeao:
                        try:
                            client = MlflowClient(tracking_uri=self._tracking_uri)
                            client.set_tag(run_id_campeao, "status_modelo", "campeao")
                            client.set_tag(run_id_campeao, "campeao", "true")
                            if "metricas_holdout" in payload and isinstance(payload["metricas_holdout"], dict):
                                for k, v in payload["metricas_holdout"].items():
                                    if isinstance(v, (int, float)):
                                        client.log_metric(run_id_campeao, f"holdout_{k}", float(v))
                        except Exception:
                            pass

                case TipoEvento.METRICAS_NEGOCIO_FINALIZADAS:
                    self._garantir_run_pai()
                    mlflow.set_tag("metricas_negocio", "concluidas")

                    nome_campeao = str(payload.get("nome_campeao") or self._nome_campeao or "")
                    run_id_campeao = self._runs_modelos.get(nome_campeao) if nome_campeao else None
                    client: MlflowClient | None = None
                    if run_id_campeao:
                        try:
                            client = MlflowClient(tracking_uri=self._tracking_uri)
                        except Exception:
                            pass

                    if "metricas_consolidadas" in payload and isinstance(payload["metricas_consolidadas"], dict):
                        metricas_dict = payload["metricas_consolidadas"]
                        mlflow.log_dict(
                            metricas_dict,
                            "negocio/metricas_imobiliaria.json",
                        )

                        c5 = float(metricas_dict.get("cobertura_5", 0.0))
                        c10 = float(metricas_dict.get("cobertura_10", 0.0))
                        c15 = float(metricas_dict.get("cobertura_15", 0.0))
                        desc_seg = float(metricas_dict.get("desconto_seguro", 0.0))
                        mae_reais = float(metricas_dict.get("mae_reais", 0.0))
                        vies_med = float(metricas_dict.get("vies_medio", 0.0))
                        r_sub = float(metricas_dict.get("risco_subprecificacao", 0.0))
                        r_sup = float(metricas_dict.get("risco_superprecificacao", 0.0))
                        margem_neg = float(metricas_dict.get("margem_negociacao_estimada", 0.0))
                        rec_perd = float(metricas_dict.get("receita_potencial_perdida", 0.0))

                        resumo_executivo: dict[str, str] = {
                            "status_modelo_negocio": "aprovado",
                            "desconto_seguro_recomendado": f"{desc_seg:.2f}%",
                            "cobertura_tolerancia_10pct": f"{c10 * 100:.2f}%",
                            "cobertura_tolerancia_5pct": f"{c5 * 100:.2f}%",
                            "cobertura_tolerancia_15pct": f"{c15 * 100:.2f}%",
                            "mae_financeiro_reais": f"R$ {mae_reais:,.2f}",
                            "vies_medio_precificacao": f"R$ {vies_med:,.2f}",
                            "risco_subprecificacao_severa": f"{r_sub * 100:.2f}%",
                            "risco_superprecificacao_severa": f"{r_sup * 100:.2f}%",
                            "margem_negociacao_estimada": f"R$ {margem_neg:,.2f}",
                            "receita_potencial_perdida": f"R$ {rec_perd:,.2f}",
                        }
                        mlflow.log_dict(resumo_executivo, "negocio/resumo_executivo_negocio.json")

                        tags_negocio: dict[str, str] = {
                            "negocio_desconto_seguro": f"{desc_seg:.2f}%",
                            "negocio_cobertura_5": f"{c5 * 100:.2f}%",
                            "negocio_cobertura_10": f"{c10 * 100:.2f}%",
                            "negocio_cobertura_15": f"{c15 * 100:.2f}%",
                            "negocio_mae_reais": f"R$ {mae_reais:,.2f}",
                            "negocio_vies_medio": f"R$ {vies_med:,.2f}",
                            "negocio_risco_subprecificacao": f"{r_sub * 100:.2f}%",
                            "negocio_risco_superprecificacao": f"{r_sup * 100:.2f}%",
                        }
                        mlflow.set_tags(tags_negocio)

                        for k, v in metricas_dict.items():
                            if isinstance(v, (int, float)):
                                val = float(v)
                                mlflow.log_metric(f"negocio_{k}", val)
                                if k not in ("mae", "rmse", "r2"):
                                    mlflow.log_metric(k, val)

                                if client and run_id_campeao:
                                    try:
                                        client.log_metric(run_id_campeao, f"negocio_{k}", val)
                                        if k not in ("mae", "rmse", "r2"):
                                            client.log_metric(run_id_campeao, k, val)
                                    except Exception:
                                        pass

                        if client and run_id_campeao:
                            for tk, tv in tags_negocio.items():
                                try:
                                    client.set_tag(run_id_campeao, tk, tv)
                                except Exception:
                                    pass

                        try:
                            fig_cob = self._gerar_figura_coberturas(c5, c10, c15)
                            mlflow.log_figure(fig_cob, "negocio/grafico_cobertura_tolerancia.png")
                            plt.close(fig_cob)
                        except Exception:
                            pass

                    if "tabela_erro_faixa" in payload and isinstance(payload["tabela_erro_faixa"], pd.DataFrame) and not payload["tabela_erro_faixa"].empty:
                        df_faixa = payload["tabela_erro_faixa"]
                        mlflow.log_table(
                            data=df_faixa,
                            artifact_file="negocio/metricas_por_faixa_preco.json",
                        )
                        mlflow.log_text(
                            text=df_faixa.to_csv(index=False),
                            artifact_file="negocio/metricas_por_faixa_preco.csv",
                        )

                    if "tabela_erro_zona" in payload and isinstance(payload["tabela_erro_zona"], pd.DataFrame) and not payload["tabela_erro_zona"].empty:
                        df_zona = payload["tabela_erro_zona"]
                        mlflow.log_table(
                            data=df_zona,
                            artifact_file="negocio/metricas_por_zona.json",
                        )
                        mlflow.log_text(
                            text=df_zona.to_csv(index=False),
                            artifact_file="negocio/metricas_por_zona.csv",
                        )

                    if "tabela_cobertura_tolerancia" in payload and isinstance(payload["tabela_cobertura_tolerancia"], pd.DataFrame) and not payload["tabela_cobertura_tolerancia"].empty:
                        df_cob = payload["tabela_cobertura_tolerancia"]
                        mlflow.log_table(
                            data=df_cob,
                            artifact_file="negocio/cobertura_tolerancia.json",
                        )
                        mlflow.log_text(
                            text=df_cob.to_csv(index=False),
                            artifact_file="negocio/cobertura_tolerancia.csv",
                        )

                case TipoEvento.DRIFT_DETECTADO:
                    self._garantir_run_pai(nome_padrao="monitoramento_drift")
                    tags_drift: dict[str, str] = {
                        "tipo_run": "drift",
                        "status_drift": str(payload.get("nivel", "desconhecido")),
                    }
                    mlflow.set_tags(tags_drift)
                    if "nivel" in payload:
                        mlflow.log_param("drift/nivel", str(payload["nivel"]))
                    if "relatorio_drift" in payload and isinstance(payload["relatorio_drift"], dict):
                        mlflow.log_dict(payload["relatorio_drift"], "drift/relatorio_drift.json")

                case TipoEvento.CURVA_APRENDIZADO_GERADA:
                    self._garantir_run_pai()
                    mlflow.set_tag("curva_aprendizado", "gerada")
                    if "figura_curva" in payload and isinstance(payload["figura_curva"], Figure):
                        mlflow.log_figure(payload["figura_curva"], "validacao/curva_aprendizado.png")
                    if "tabela_curva" in payload and isinstance(payload["tabela_curva"], pd.DataFrame):
                        mlflow.log_table(payload["tabela_curva"], "validacao/curva_aprendizado.json")
                    if "diagnostico" in payload and isinstance(payload["diagnostico"], dict):
                        mlflow.log_dict(payload["diagnostico"], "validacao/diagnostico_aprendizado.json")

        except Exception:
            # Em caso de indisponibilidade transitória do MLflow em testes, não interrompe o fluxo
            pass
