from pathlib import Path
from typing import Final, override

import yaml

from app_build.configuracao_sistema.contrato_configuracao import (
    CaminhoArquivo,
    ConfiguracaoAmostra,
    ConfiguracaoAvaliacao,
    ConfiguracaoDados,
    ConfiguracaoGeral,
    ConfiguracaoMlflow,
    ConfiguracaoSelecao,
    ConfiguracaoValidacao,
    ContratoConfiguracao,
)


class FalhaConfiguracaoErro(Exception):
    def __init__(self, mensagem: str, causa: Exception | None = None) -> None:
        super().__init__(mensagem)
        self.causa: Final[Exception | None] = causa


class LeitorYaml(ContratoConfiguracao):
    @override
    def carregar_geral(self, caminho_pipeline: CaminhoArquivo) -> ConfiguracaoGeral:
        try:
            caminho = Path(caminho_pipeline)
            conteudo = yaml.safe_load(caminho.read_text(encoding="utf-8"))

            secao_dados = conteudo["dados"]
            secao_cv = conteudo["validacao_cruzada"]
            secao_aval = conteudo["avaliacao"]
            secao_sel = conteudo["selecao_modelos"]
            secao_amostra = conteudo["amostra"]
            secao_mlflow = conteudo["mlflow"]

            dados = ConfiguracaoDados(
                target=str(secao_dados["target"]),
                grupos=tuple(map(str, secao_dados["grupos"])),
                proporcao_holdout=float(secao_dados["holdout"]["proporcao"]),
                semente_holdout=int(secao_dados["holdout"]["random_state"]),
            )

            validacao = ConfiguracaoValidacao(
                splits_externos=int(secao_cv["externa"]["splits"]),
                repeticoes_externas=int(secao_cv["externa"]["repeticoes"]),
                semente_externa=int(secao_cv["externa"]["random_state"]),
                splits_internos=int(secao_cv["interna"]["splits"]),
                embaralhar_interno=bool(secao_cv["interna"]["embaralhar"]),
                semente_interna=int(secao_cv["interna"]["random_state"]),
            )

            avaliacao = ConfiguracaoAvaliacao(
                metrica_principal=str(secao_aval["metrica_principal"]),
                metricas=tuple(map(str, secao_aval["metricas"])),
            )

            selecao = ConfiguracaoSelecao(
                usar_votacao=bool(secao_sel["votacao"]),
                quantidade_modelos=int(secao_sel["quantidade_modelos"]),
                criterio_unico=str(secao_sel["criterio_modelo_unico"]),
            )

            amostra = ConfiguracaoAmostra(
                minima_zona=int(secao_amostra["minima_zona"]),
                minima_bairro=int(secao_amostra["minima_bairro"]),
            )

            mlflow = ConfiguracaoMlflow(
                registrar_tuning=bool(secao_mlflow["registrar_tuning"]),
                registrar_parametros=bool(secao_mlflow["registrar_parametros"]),
                registrar_interpretacao_negocio=bool(
                    secao_mlflow["registrar_interpretacao_negocio"]
                ),
                registrar_artefatos=bool(secao_mlflow["registrar_artefatos"]),
            )

            return ConfiguracaoGeral(
                dados=dados,
                validacao=validacao,
                avaliacao=avaliacao,
                selecao=selecao,
                amostra=amostra,
                mlflow=mlflow,
            )
        except Exception as erro:
            raise FalhaConfiguracaoErro(
                f"Erro ao ler configuracao de pipeline em {caminho_pipeline}: {erro}",
                erro,
            ) from erro

    @override
    def carregar_modelos(
        self, caminho_modelos: CaminhoArquivo
    ) -> dict[str, dict[str, object]]:
        try:
            caminho = Path(caminho_modelos)
            conteudo = yaml.safe_load(caminho.read_text(encoding="utf-8"))
            modelos_brutos: dict[str, dict[str, object]] = conteudo["modelos"]
            return modelos_brutos
        except Exception as erro:
            raise FalhaConfiguracaoErro(
                f"Erro ao ler configuracao de modelos em {caminho_modelos}: {erro}",
                erro,
            ) from erro
