import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Final, override

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models.signature import infer_signature

from app_build.ajuste_modelos.explicador_parametros import ExplicadorParametros
from app_build.observabilidade_metricas.coletor_prometheus import ColetorPrometheus
from app_build.rastreamento_mlflow.contrato_observador import (
    ContratoObservador,
    EventoEstatisticaConcluida,
    EventoHoldoutAvaliado,
    EventoModeloSelecionado,
    EventoNestedCvConcluido,
    EventoRegrasNegocioConcluido,
    EventoTreinoFinalConcluido,
)
from app_build.rastreamento_mlflow.empacotador_modelo import EmpacotadorModelo


class ObservadorMlflow(ContratoObservador):
    def __init__(
        self,
        nome_experimento: str = "previsao_preco_apartamentos_ribeirao_preto",
        tracking_uri: str | None = None,
        coletor: ColetorPrometheus | None = None,
    ) -> None:
        uri_env = os.getenv("MLFLOW_TRACKING_URI")
        uri = tracking_uri or (str(uri_env) if uri_env else "http://localhost:5000")
        self._tracking_uri: Final[str] = uri
        self._nome_experimento: Final[str] = nome_experimento
        self._explicador: Final[ExplicadorParametros] = ExplicadorParametros()
        self._eventos_regras: Final[list[EventoRegrasNegocioConcluido]] = []
        self._runs_regras: Final[list[str]] = []
        self._coletor: Final[ColetorPrometheus | None] = coletor

    def inicializar(self) -> None:
        try:
            mlflow.set_tracking_uri(self._tracking_uri)
            mlflow.set_experiment(self._nome_experimento)
        except (MlflowException, OSError, ValueError) as erro:
            sys.stderr.write(
                f"Aviso: Nao foi possivel conectar ao servidor MLflow ({erro}). Modo offline ativo.\n"
            )

    @override
    def ao_concluir_nested_cv(self, evento: EventoNestedCvConcluido) -> None:
        try:
            with mlflow.start_run(run_name=f"nested_cv_{evento.nome_modelo}"):
                mlflow.set_tag("etapa", "nested_cv")
                mlflow.set_tag("modelo", evento.nome_modelo)
                mlflow.log_param("desvio_padrao_ddof", 0)
                mlflow.log_param("total_folds_externos", len(evento.resultado_cv.resultados_folds))
                mlflow.log_metrics({
                    f"{nome}_std": valor
                    for nome, valor in asdict(evento.resultado_cv.metricas_desvios_padrao).items()
                })
                mlflow.log_metrics({
                    f"{nome}_medio": valor
                    for nome, valor in asdict(evento.resultado_cv.metricas_medias).items()
                })

                mlflow.log_metric(
                    "rmse_mediano", evento.resultado_cv.metricas_medianas.rmse
                )
                mlflow.log_metric(
                    "mae_mediano", evento.resultado_cv.metricas_medianas.mae
                )

                md_conteudo = self._explicador.gerar_markdown(
                    evento.nome_modelo, evento.explicacoes_parametros
                )
                caminho_artefato = Path(f"/tmp/explicabilidade_{evento.nome_modelo}.md")
                caminho_artefato.write_text(md_conteudo, encoding="utf-8")
                mlflow.log_artifact(
                    str(caminho_artefato), artifact_path="explicabilidade"
                )

                for fold in evento.resultado_cv.resultados_folds:
                    with mlflow.start_run(
                        run_name=f"fold_{fold.indice_fold}", nested=True
                    ):
                        mlflow.set_tag("fold", str(fold.indice_fold))
                        mlflow.log_metric("rmse", fold.metricas.rmse)
                        mlflow.log_metric("mae", fold.metricas.mae)
                        mlflow.log_metric("r2", fold.metricas.r2)
                        mlflow.log_metric("duracao_s", fold.tempo_segundos)
                        for param_chave, param_val in fold.melhores_parametros.items():
                            mlflow.log_param(f"best_{param_chave}", str(param_val))
        except (MlflowException, OSError, ValueError, RuntimeError) as erro:
            sys.stderr.write(f"Aviso de rastreamento MLflow em nested_cv: {erro}\n")

    @override
    def ao_concluir_estatistica(self, evento: EventoEstatisticaConcluida) -> None:
        try:
            with mlflow.start_run(run_name="avaliacao_estatistica"):
                mlflow.set_tag("etapa", "estatistica_comparativa")
                mlflow.log_metric(
                    "friedman_estatistica", evento.resultado_friedman.estatistica
                )
                mlflow.log_metric("friedman_p_valor", evento.resultado_friedman.p_valor)
                mlflow.log_metric(
                    "nemenyi_cd", evento.resultado_nemenyi.diferenca_critica
                )

                # ── JSON para consumo programático ──────────────────────────────
                dados_ranks = {
                    "ranks_friedman": evento.resultado_friedman.ranks_medios,
                    "shapiro": {
                        s.nome_modelo: {"p_valor": s.p_valor, "normal": s.eh_normal}
                        for s in evento.resultados_shapiro
                    },
                }
                caminho_json = Path("/tmp/resumo_estatistico.json")
                caminho_json.write_text(
                    json.dumps(dados_ranks, indent=2), encoding="utf-8"
                )
                mlflow.log_artifact(str(caminho_json), artifact_path="estatistica")

                # ── Relatório Markdown com Tabelas ──────────────────────────────
                friedman = evento.resultado_friedman
                nemenyi = evento.resultado_nemenyi
                cd = nemenyi.diferenca_critica
                sig_friedman = "✅ Sim" if friedman.eh_significativo else "❌ Não"

                # Tabela 1 — Friedman
                linhas_md: list[str] = [
                    "# Relatório de Testes Estatísticos — Comparação de Modelos",
                    "",
                    "## 1. Teste de Friedman (Comparação Global)",
                    "",
                    "| Métrica | Valor |",
                    "| :--- | ---: |",
                    f"| Estatística χ² de Friedman | `{friedman.estatistica:.4f}` |",
                    f"| P-Valor | `{friedman.p_valor:.6f}` |",
                    f"| Diferença Estatisticamente Significativa (α = 0.05) | {sig_friedman} |",
                    "",
                ]

                # Tabela 2 — Ranks Médios por Modelo
                linhas_md += [
                    "## 2. Ranks Médios por Modelo (Friedman)",
                    "",
                    "> Menor rank = melhor desempenho relativo no conjunto de validação cruzada.",
                    "",
                    "| # | Modelo | Rank Médio |",
                    "| :---: | :--- | ---: |",
                ]
                ranks_ordenados = sorted(
                    friedman.ranks_medios.items(), key=lambda x: x[1]
                )
                for pos, (modelo, rank) in enumerate(ranks_ordenados, start=1):
                    medalha = {1: "🥇", 2: "🥈", 3: "🥉"}.get(pos, f"{pos}º")
                    linhas_md.append(f"| {medalha} | `{modelo}` | `{rank:.4f}` |")
                linhas_md.append("")

                # Tabela 3 — Comparações Pareadas Nemenyi
                linhas_md += [
                    f"## 3. Comparações Pareadas — Teste Post-Hoc de Nemenyi (CD = {cd:.4f})",
                    "",
                    "> **CD** = Diferença Crítica. Se `|ΔRank| > CD`, a diferença é estatisticamente significativa.",
                    "",
                    "| Modelo A | Modelo B | ΔRank | CD | Significativo? |",
                    "| :--- | :--- | ---: | ---: | :---: |",
                ]
                for comp in sorted(
                    nemenyi.comparacoes, key=lambda c: abs(c.diferenca_ranks), reverse=True
                ):
                    sig = "✅ Sim" if comp.eh_significativo else "—"
                    linhas_md.append(
                        f"| `{comp.modelo_a}` | `{comp.modelo_b}` "
                        f"| `{comp.diferenca_ranks:.4f}` "
                        f"| `{comp.diferenca_critica:.4f}` "
                        f"| {sig} |"
                    )
                linhas_md.append("")

                # Tabela 4 — Shapiro-Wilk por Modelo
                linhas_md += [
                    "## 4. Normalidade dos Resíduos — Teste Shapiro-Wilk",
                    "",
                    "> P-valor > 0.05 indica que os resíduos seguem distribuição normal.",
                    "",
                    "| Modelo | Estatística W | P-Valor | Normal? |",
                    "| :--- | ---: | ---: | :---: |",
                ]
                for s in sorted(evento.resultados_shapiro, key=lambda x: x.nome_modelo):
                    normal = "✅ Sim" if s.eh_normal else "❌ Não"
                    linhas_md.append(
                        f"| `{s.nome_modelo}` "
                        f"| `{s.estatistica:.4f}` "
                        f"| `{s.p_valor:.6f}` "
                        f"| {normal} |"
                    )
                linhas_md.append("")

                # ── Salvar e logar artifact ─────────────────────────────────────
                caminho_md = Path("/tmp/relatorio_testes_estatisticos.md")
                caminho_md.write_text("\n".join(linhas_md), encoding="utf-8")
                mlflow.log_artifact(str(caminho_md), artifact_path="estatistica")

        except (MlflowException, OSError, ValueError, RuntimeError) as erro:
            sys.stderr.write(f"Aviso de rastreamento MLflow em estatistica: {erro}\n")

        # ── Prometheus: publica métricas dos testes estatísticos ────────────────
        if self._coletor is not None:
            try:
                friedman = evento.resultado_friedman
                nemenyi = evento.resultado_nemenyi
                self._coletor.registrar_testes_estatisticos(
                    friedman_chi2=friedman.estatistica,
                    friedman_pvalor=friedman.p_valor,
                    friedman_significativo=friedman.eh_significativo,
                    nemenyi_cd=nemenyi.diferenca_critica,
                    ranks_modelos=friedman.ranks_medios,
                    comparacoes_nemenyi=[
                        (c.modelo_a, c.modelo_b, c.diferenca_ranks, c.eh_significativo)
                        for c in nemenyi.comparacoes
                    ],
                    resultados_shapiro={
                        s.nome_modelo: (s.p_valor, s.eh_normal)
                        for s in evento.resultados_shapiro
                    },
                )
            except Exception as err:
                sys.stderr.write(f"Aviso Prometheus em estatistica: {err}\n")


    @override
    def ao_selecionar_modelo(self, evento: EventoModeloSelecionado) -> None:
        try:
            with mlflow.start_run(run_name="selecao_modelo_final"):
                mlflow.set_tag("etapa", "selecao")
                mlflow.set_tag("modelo_campeao", evento.decisao.nome_modelo_final)
                mlflow.set_tag("modo_selecao", evento.decisao.modo_selecao)
                mlflow.log_param("justificativa", evento.decisao.justificativa)
        except (MlflowException, OSError, ValueError, RuntimeError) as erro:
            sys.stderr.write(f"Aviso de rastreamento MLflow em selecao: {erro}\n")

    @override
    def ao_concluir_treino_final(self, evento: EventoTreinoFinalConcluido) -> None:
        try:
            with mlflow.start_run(run_name=f"modelo_final_{evento.nome_modelo}"):
                mlflow.set_tag("etapa", "treino_final")
                mlflow.set_tag("modelo", evento.nome_modelo)
                mlflow.log_dict(
                    evento.parametros_componentes,
                    "treino_final/parametros_componentes.json",
                )
                mlflow.log_text(
                    self._explicador.gerar_markdown(
                        evento.nome_modelo, evento.explicacoes_parametros
                    ),
                    "treino_final/interpretacao_parametros.md",
                )
                mlflow.set_tag("regras_negocio_hierarquia", "GLOBAL_ZONA_BAIRRO")
                mlflow.set_tag(
                    "simulador_descontos", "moderado_5_agressivo_10_queima_15"
                )
                mlflow.set_tag("municipio", "Ribeirao_Preto_SP")

                for run_id_regras in self._runs_regras:
                    mlflow.set_tag("run_regras_negocio_id", run_id_regras)

                for ev_regras in self._eventos_regras:
                    estat_glob = ev_regras.estatisticas_hierarquicas.estatisticas_global
                    mlflow.log_metric(
                        "regras_global_media_valor", estat_glob.media_valor
                    )
                    mlflow.log_metric(
                        "regras_global_mediana_m2", estat_glob.mediana_valor_m2
                    )
                    for k, v in ev_regras.metricas_negocio.items():
                        mlflow.log_metric(f"regras_{k}", v)

                    caminho_dir = Path("/tmp/regras_negocio")
                    mlflow.log_artifacts(
                        str(caminho_dir), artifact_path="regras_negocio"
                    )

                colunas_indesejadas_api: tuple[str, ...] = (
                    "Código",
                    "Apartamento",
                    "valor_m2",
                    "media_valor_m2_bairro",
                    "media_valor_m2_zona",
                )
                cols_exemplo = set(evento.dados_exemplo.columns)
                cols_drop_api: tuple[str, ...] = tuple(
                    filter(cols_exemplo.__contains__, colunas_indesejadas_api)
                )
                dados_exemplo_api = evento.dados_exemplo.drop(
                    columns=list(cols_drop_api)
                )

                modelo_envelopado = EmpacotadorModelo(
                    pipeline_scikit=evento.estimador,
                    motor_imobiliario=evento.motor_imobiliario,
                )
                exemplo_x = dados_exemplo_api.head(5)
                previsoes_exemplo = modelo_envelopado.predict(
                    context=None, model_input=exemplo_x
                )

                # ── Assinatura explícita — sem Código/Apartamento ───────────────
                from mlflow.models.signature import ModelSignature
                from mlflow.types.schema import ColSpec, Schema

                assinatura = ModelSignature(
                    inputs=Schema([
                        ColSpec("string", "Bairro"),
                        ColSpec("string", "Zona"),
                        ColSpec("long",   "Quartos"),
                        ColSpec("long",   "Banheiros"),
                        ColSpec("long",   "Vagas_Garagem"),
                        ColSpec("double", "Metragem"),
                    ]),
                    outputs=infer_signature(exemplo_x, previsoes_exemplo).outputs,
                )


                info_modelo = mlflow.pyfunc.log_model(
                    artifact_path="modelo",
                    python_model=modelo_envelopado,
                    signature=assinatura,
                    input_example=exemplo_x,
                    registered_model_name="previsao_preco_apartamento_modelo",
                )
                versao = str(info_modelo.registered_model_version or "1")
                cliente = mlflow.tracking.MlflowClient(tracking_uri=self._tracking_uri)
                cliente.set_registered_model_alias(
                    name="previsao_preco_apartamento_modelo",
                    alias="champion",
                    version=versao,
                )
                cliente.set_registered_model_tag(
                    name="previsao_preco_apartamento_modelo",
                    key="regras_negocio_hierarquia",
                    value="GLOBAL_ZONA_BAIRRO",
                )
                cliente.set_registered_model_tag(
                    name="previsao_preco_apartamento_modelo",
                    key="simulador_descontos",
                    value="moderado_5_agressivo_10_queima_15",
                )
                cliente.set_model_version_tag(
                    name="previsao_preco_apartamento_modelo",
                    version=versao,
                    key="regras_negocio",
                    value="GLOBAL_ZONA_BAIRRO",
                )
                cliente.set_model_version_tag(
                    name="previsao_preco_apartamento_modelo",
                    version=versao,
                    key="simulador_descontos",
                    value="moderado_5_agressivo_10_queima_15",
                )
                cliente.set_model_version_tag(
                    name="previsao_preco_apartamento_modelo",
                    version=versao,
                    key="municipio",
                    value="Ribeirao_Preto_SP",
                )
                cliente.set_model_version_tag(
                    name="previsao_preco_apartamento_modelo",
                    version=versao,
                    key="status_regras_negocio",
                    value="INTEGRADO",
                )
        except (MlflowException, OSError, ValueError, RuntimeError) as erro:
            sys.stderr.write(f"Aviso de rastreamento MLflow em treino final: {erro}\n")

    @override
    def ao_avaliar_holdout(self, evento: EventoHoldoutAvaliado) -> None:
        try:
            with mlflow.start_run(run_name="avaliacao_holdout_final"):
                mlflow.set_tag("etapa", "holdout_final")
                mlflow.log_metric("holdout_rmse_global", evento.metricas_global.rmse)
                mlflow.log_metric("holdout_mae_global", evento.metricas_global.mae)
                mlflow.log_metric("holdout_r2_global", evento.metricas_global.r2)
                mlflow.log_metric("holdout_mape_global", evento.metricas_global.mape)

                for zona, m in evento.metricas_zona.items():
                    mlflow.log_metric(f"holdout_rmse_zona_{zona}", m.rmse)
        except (MlflowException, OSError, ValueError, RuntimeError) as erro:
            sys.stderr.write(f"Aviso de rastreamento MLflow em holdout: {erro}\n")

        # ── Prometheus: métricas do modelo no holdout ───────────────────────────
        if self._coletor is not None:
            try:
                m = evento.metricas_global
                self._coletor.atualizar_metricas_modelo_completo(
                    rmse=m.rmse,
                    mae=m.mae,
                    r2=m.r2,
                    mape=m.mape,
                    total_amostras_treino=0,   # preenchido em ao_concluir_treino_final
                    total_amostras_holdout=len(evento.metricas_bairro),
                    versao_modelo="holdout",
                    nome_modelo="campeao",
                )
            except Exception as err:
                sys.stderr.write(f"Aviso Prometheus em holdout: {err}\n")


    @override
    def ao_concluir_regras_negocio(
        self, evento: EventoRegrasNegocioConcluido
    ) -> None:
        try:
            self._eventos_regras.append(evento)
            with mlflow.start_run(run_name="regras_negocio_imobiliarias") as run:
                self._runs_regras.append(run.info.run_id)
                mlflow.set_tag("etapa", "regras_negocio")
                mlflow.set_tag("hierarquia", "GLOBAL_ZONA_BAIRRO")
                mlflow.set_tag("municipio", "Ribeirao_Preto_SP")
                mlflow.set_tag(
                    "simulador_descontos", "moderado_5_agressivo_10_queima_15"
                )

                estat_glob = evento.estatisticas_hierarquicas.estatisticas_global
                mlflow.log_metric("regras_global_media_valor", estat_glob.media_valor)
                mlflow.log_metric(
                    "regras_global_mediana_valor", estat_glob.mediana_valor
                )
                mlflow.log_metric(
                    "regras_global_media_m2", estat_glob.media_valor_m2
                )
                mlflow.log_metric(
                    "regras_global_mediana_m2", estat_glob.mediana_valor_m2
                )
                mlflow.log_metric(
                    "regras_global_desvio_m2", estat_glob.desvio_valor_m2
                )

                for (
                    zona,
                    estat_zona,
                ) in evento.estatisticas_hierarquicas.tabela_zonas.items():
                    zona_chave = zona.replace(" ", "_").lower()
                    mlflow.log_metric(
                        f"regras_m2_mediana_{zona_chave}",
                        estat_zona.mediana_valor_m2,
                    )
                    mlflow.log_metric(
                        f"regras_m2_media_{zona_chave}", estat_zona.media_valor_m2
                    )

                for chave_metrica, valor_metrica in evento.metricas_negocio.items():
                    mlflow.log_metric(f"regras_{chave_metrica}", valor_metrica)

                caminho_dir = Path("/tmp/regras_negocio")
                caminho_dir.mkdir(parents=True, exist_ok=True)

                dados_zonas = {
                    z: {
                        "media_m2": e.media_valor_m2,
                        "mediana_m2": e.mediana_valor_m2,
                        "desvio_m2": e.desvio_valor_m2,
                        "total_amostras": e.total_amostras,
                        "status": e.status_amostral.value,
                    }
                    for z, e in evento.estatisticas_hierarquicas.tabela_zonas.items()
                }
                caminho_zonas = caminho_dir / "tabela_zonas.json"
                caminho_zonas.write_text(
                    json.dumps(dados_zonas, indent=2), encoding="utf-8"
                )
                mlflow.log_artifact(str(caminho_zonas), artifact_path="regras_negocio")

                dados_bairros = {
                    b: {
                        "media_m2": e.media_valor_m2,
                        "mediana_m2": e.mediana_valor_m2,
                        "total_amostras": e.total_amostras,
                        "status": e.status_amostral.value,
                    }
                    for b, e in evento.estatisticas_hierarquicas.tabela_bairros.items()
                }
                caminho_bairros = caminho_dir / "tabela_bairros.json"
                caminho_bairros.write_text(
                    json.dumps(dados_bairros, indent=2), encoding="utf-8"
                )
                mlflow.log_artifact(
                    str(caminho_bairros), artifact_path="regras_negocio"
                )

                caminho_exemplos = caminho_dir / "exemplos_simulacao_descontos.json"
                dados_exemplos = [asdict(e) for e in evento.exemplos_simulacao]
                caminho_exemplos.write_text(
                    json.dumps(dados_exemplos, default=str, indent=2),
                    encoding="utf-8",
                )
                mlflow.log_artifact(
                    str(caminho_exemplos), artifact_path="regras_negocio"
                )

                caminho_csv = caminho_dir / "holdout_enriquecido.csv"
                evento.dados_enriquecidos.to_csv(caminho_csv, index=False)
                mlflow.log_artifact(str(caminho_csv), artifact_path="regras_negocio")

                linhas_md = [
                    "# Relatório de Regras de Negócio e Precificação Imobiliária",
                    "## Município: Ribeirão Preto / SP",
                    "",
                    "### 1. Parâmetros Globais",
                    f"- **Preço Médio:** R$ {estat_glob.media_valor:,.2f}",
                    f"- **Preço Mediano:** R$ {estat_glob.mediana_valor:,.2f}",
                    f"- **Valor Médio do m²:** R$ {estat_glob.media_valor_m2:,.2f}",
                    f"- **Valor Mediano do m²:** R$ {estat_glob.mediana_valor_m2:,.2f}",
                    "",
                    "### 2. Estratégias Comerciais e Liquidez",
                    "- Desconto Moderado (Margem Negociação): 5%",
                    "- Desconto Agressivo (Liquidez Rápida): 10%",
                    "- Desconto Queima (Oportunidade/Liquidação): 15%",
                    "- Faixa Segura de Preço: 90% a 100% do valor avaliado",
                ]
                caminho_relatorio = caminho_dir / "relatorio_regras_negocio.md"
                caminho_relatorio.write_text("\n".join(linhas_md), encoding="utf-8")
                mlflow.log_artifact(
                    str(caminho_relatorio), artifact_path="regras_negocio"
                )
        except (MlflowException, OSError, ValueError, RuntimeError) as erro:
            sys.stderr.write(
                f"Aviso de rastreamento MLflow em regras de negocio: {erro}\n"
            )

        # ── Prometheus: métricas de negócio hierárquico ─────────────────────────
        if self._coletor is not None:
            try:
                hier = evento.estatisticas_hierarquicas
                glob = hier.estatisticas_global

                self._coletor.registrar_metricas_negocio(
                    preco_mediano_global=glob.mediana_valor,
                    preco_medio_global=glob.media_valor,
                    preco_m2_mediano_global=glob.mediana_valor_m2,
                    total_imoveis_dataset=glob.total_amostras,
                )

                for zona, estat in hier.tabela_zonas.items():
                    self._coletor.registrar_metricas_negocio_zona(
                        zona=zona,
                        preco_mediano=estat.mediana_valor,
                        preco_m2_mediano=estat.mediana_valor_m2,
                        total_amostras=estat.total_amostras,
                    )

                # Determina a zona de cada bairro via dados enriquecidos
                df = evento.dados_enriquecidos
                bairro_zona: dict[str, str] = {}
                if "Bairro" in df.columns and "Zona" in df.columns:
                    bairro_zona = (
                        df[["Bairro", "Zona"]]
                        .drop_duplicates()
                        .set_index("Bairro")["Zona"]
                        .to_dict()
                    )

                for bairro, estat in hier.tabela_bairros.items():
                    self._coletor.registrar_metricas_negocio_bairro(
                        bairro=bairro,
                        zona=bairro_zona.get(bairro, "Desconhecida"),
                        preco_mediano=estat.mediana_valor,
                        total_amostras=estat.total_amostras,
                    )
            except Exception as err:
                sys.stderr.write(f"Aviso Prometheus em regras de negocio: {err}\n")
