import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class ExplicacaoParametro:
    parametro: str
    valor: object
    interpretacao_negocio: str


DICIONARIO_EXPLICACOES: Final[Mapping[str, str]] = {
    "alpha": "Intensidade de regularizacao penalizando coeficientes extremos; valores maiores aumentam a estabilidade contra variabilidade espuria de precos.",
    "l1_ratio": "Proporcao de selecao direta de atributos (L1) versus suavizacao de pesos (L2), equilibrando simplicidade e dispersao.",
    "max_depth": "Profundidade permitida na arvore; controla o trade-off entre regras de precificacao simples e subdivisoes muito especificas por imovel.",
    "min_samples_split": "Volume minimo de imoveis necessarios para ramificar uma regra de mercado imobiliario.",
    "min_samples_leaf": "Numero minimo de apartamentos contidos em cada precificacao terminal (folha).",
    "n_estimators": "Quantidade de avaliadores combinados no comite; valores maiores aumentam o consenso e reduzem a variancia.",
    "max_features": "Proporcao de atributos construtivos considerados em cada divisao para diversificar os consensos das arvores.",
    "C": "Margem de tolerancia a imoveis com precos atipicos (outliers) no ajuste do hiperplano de suporte.",
    "epsilon": "Zona de indiferenca onde pequenas divergencias de preco nao geram penalizacao ao estimador.",
    "gamma": "Raio de influencia espacial e nao-linear de cada imovel de referencia no kernel.",
    "kernel": "Tipo de transformacao dimensional usada para captar relacoes entre metragens, bairros e precos.",
    "hidden_layer_sizes": "Capacidade e profundidade representacional da rede neural para capturar interacoes entre atributos construtivos.",
    "learning_rate": "Taxa de aprendizado determinando a cautela das correcoes iterativas a cada arvore ou epoca de treinamento.",
    "num_leaves": "Complexidade estrutural de cada arvore no gradient boosting, limitando a expressividade das regras.",
    "subsample": "Percentual de imoveis sorteados a cada rodada de boosting para evitar superajuste aos maiores apartamentos.",
    "colsample_bytree": "Percentual de caracteristicas sorteadas por arvore no XGBoost para evitar dependencia excessiva de uma unica variavel.",
    "solver": "Algoritmo matematico de convergencia e resolucao da regressao regularizada.",
    "fit_intercept": "Calculo do preco base estrutural (intercepto) quando todas as dimensoes fisicas sao nulas.",
}


class ExplicadorParametros:
    def explicar(
        self, nome_modelo: str, parametros_efetivos: Mapping[str, object]
    ) -> tuple[ExplicacaoParametro, ...]:
        def mapear_param(item: tuple[str, object]) -> ExplicacaoParametro:
            chave, val = item
            chave_limpa = chave.replace("modelo__", "")
            texto = DICIONARIO_EXPLICACOES.get(
                chave_limpa,
                f"Configuracao tecnica operacional '{chave_limpa}' calibrada para estabilidade numerica.",
            )
            return ExplicacaoParametro(
                parametro=chave_limpa,
                valor=val,
                interpretacao_negocio=texto,
            )

        return tuple(map(mapear_param, parametros_efetivos.items()))

    def gerar_markdown(
        self, nome_modelo: str, explicacoes: tuple[ExplicacaoParametro, ...]
    ) -> str:
        linhas = [
            f"# Interpretacao de Negocio dos Parametros — {nome_modelo}",
            "",
            "| Parametro | Valor Utilizado | Interpretacao para Negocio Imobiliario |",
            "|---|---|---|",
        ]
        for exp in explicacoes:
            valor_str = json.dumps(exp.valor)
            linhas.append(
                f"| `{exp.parametro}` | `{valor_str}` | {exp.interpretacao_negocio} |"
            )
        linhas.append("")
        return "\n".join(linhas)
