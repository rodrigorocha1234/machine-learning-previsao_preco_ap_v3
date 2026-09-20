"""Observador concreto para envio de métricas, figuras e artefatos ao MLflow."""

import os
from matplotlib.figure import Figure
import mlflow
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
    ) -> None:
        self._tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self._nome_experimento = nome_experimento
        self._configurar_mlflow()

    def _configurar_mlflow(self) -> None:
        """Define o Tracking URI e o experimento ativo."""
        try:
            mlflow.set_tracking_uri(self._tracking_uri)
            mlflow.set_experiment(self._nome_experimento)
        except Exception:
            # Permite execução isolada se o servidor estiver offline
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
                    for k, v in payload.items():
                        if isinstance(v, pd.DataFrame):
                            mlflow.log_table(data=v, artifact_file=f"eda/{k}.json")
                        elif isinstance(v, Figure):
                            mlflow.log_figure(figure=v, artifact_file=f"eda/{k}.png")
                        elif isinstance(v, dict):
                            mlflow.log_dict(dictionary=v, artifact_file=f"eda/{k}.json")

                case TipoEvento.GRIDSEARCH_INICIADO:
                    nome = str(payload.get("nome_modelo", "modelo"))
                    mlflow.log_param(f"gridsearch_{nome}_iniciado", True)

                case TipoEvento.GRIDSEARCH_FINALIZADO:
                    nome = str(payload.get("nome_modelo", "modelo"))
                    if "melhores_parametros" in payload and isinstance(payload["melhores_parametros"], dict):
                        mlflow.log_dict(
                            dictionary=payload["melhores_parametros"],
                            artifact_file=f"gridsearch/{nome}_melhores_parametros.json",
                        )
                    if "melhor_score" in payload and isinstance(payload["melhor_score"], (int, float)):
                        mlflow.log_metric(f"gridsearch_{nome}_best_rmse", float(payload["melhor_score"]))
                    if "tabela_cv_results" in payload and isinstance(payload["tabela_cv_results"], pd.DataFrame):
                        mlflow.log_table(
                            data=payload["tabela_cv_results"],
                            artifact_file=f"gridsearch/{nome}_cv_results.json",
                        )

                case TipoEvento.VALIDACAO_FINALIZADA:
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

                case TipoEvento.FRIEDMAN_FINALIZADO:
                    if "friedman" in payload and isinstance(payload["friedman"], dict):
                        mlflow.log_dict(payload["friedman"], "estatistica/friedman.json")
                    if "p_valor" in payload and isinstance(payload["p_valor"], (int, float)):
                        mlflow.log_metric("friedman_p_valor", float(payload["p_valor"]))
                    if "estatistica" in payload and isinstance(payload["estatistica"], (int, float)):
                        mlflow.log_metric("friedman_estatistica", float(payload["estatistica"]))

                case TipoEvento.NEMENYI_FINALIZADO:
                    if "matriz_p_valores" in payload and isinstance(payload["matriz_p_valores"], pd.DataFrame):
                        mlflow.log_table(
                            data=payload["matriz_p_valores"],
                            artifact_file="estatistica/nemenyi_pvalores.json",
                        )

                case TipoEvento.ENSEMBLE_FINALIZADO:
                    mlflow.log_param("ensemble/usado", bool(payload.get("usado", False)))
                    if "estrategia" in payload:
                        mlflow.log_param("ensemble/estrategia", str(payload["estrategia"]))
                    if "pesos" in payload and isinstance(payload["pesos"], dict):
                        mlflow.log_dict(payload["pesos"], "ensemble/pesos.json")

                case TipoEvento.MODELO_CAMPEAO:
                    if "nome_campeao" in payload:
                        mlflow.log_param("campeao/nome", str(payload["nome_campeao"]))
                    if "metricas_holdout" in payload and isinstance(payload["metricas_holdout"], dict):
                        for k, v in payload["metricas_holdout"].items():
                            if isinstance(v, (int, float)):
                                mlflow.log_metric(f"holdout_{k}", float(v))

                case TipoEvento.METRICAS_NEGOCIO_FINALIZADAS:
                    if "metricas_consolidadas" in payload and isinstance(payload["metricas_consolidadas"], dict):
                        mlflow.log_dict(
                            payload["metricas_consolidadas"],
                            "negocio/metricas_imobiliaria.json",
                        )
                        for k, v in payload["metricas_consolidadas"].items():
                            if isinstance(v, (int, float)):
                                mlflow.log_metric(f"negocio_{k}", float(v))
                    if "tabela_erro_faixa" in payload and isinstance(payload["tabela_erro_faixa"], pd.DataFrame):
                        mlflow.log_table(
                            payload["tabela_erro_faixa"],
                            "negocio/metricas_por_faixa_preco.json",
                        )
                    if "tabela_erro_zona" in payload and isinstance(payload["tabela_erro_zona"], pd.DataFrame):
                        mlflow.log_table(
                            payload["tabela_erro_zona"],
                            "negocio/metricas_por_zona.json",
                        )

                case TipoEvento.DRIFT_DETECTADO:
                    if "nivel" in payload:
                        mlflow.log_param("drift/nivel", str(payload["nivel"]))
                    if "relatorio_drift" in payload and isinstance(payload["relatorio_drift"], dict):
                        mlflow.log_dict(payload["relatorio_drift"], "drift/relatorio_drift.json")

                case TipoEvento.CURVA_APRENDIZADO_GERADA:
                    if "figura_curva" in payload and isinstance(payload["figura_curva"], Figure):
                        mlflow.log_figure(payload["figura_curva"], "validacao/curva_aprendizado.png")
                    if "tabela_curva" in payload and isinstance(payload["tabela_curva"], pd.DataFrame):
                        mlflow.log_table(payload["tabela_curva"], "validacao/curva_aprendizado.json")
                    if "diagnostico" in payload and isinstance(payload["diagnostico"], dict):
                        mlflow.log_dict(payload["diagnostico"], "validacao/diagnostico_aprendizado.json")

        except Exception:
            # Em caso de indisponibilidade transitória do MLflow em testes, não interrompe o fluxo
            pass
