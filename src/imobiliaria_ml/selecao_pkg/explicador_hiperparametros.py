"""Tradutor de hiperparâmetros técnicos para linguagem de negócio."""


class ExplicadorHiperparametros:
    """Traduz os hiperparâmetros selecionados pelo Grid Search para linguagem acessível à equipe de negócio.

    Para cada parâmetro selecionado, fornece:
    - Explicação em linguagem comercial
    - Impacto de aumentar o valor
    - Impacto de diminuir o valor
    - Interpretação contextualizada do valor escolhido
    """

    # -------------------------------------------------------------------------
    # Catálogo completo de explicações de negócio por parâmetro
    # Chave: nome do parâmetro SEM prefixo "modelo__"
    # -------------------------------------------------------------------------
    _CATALOGO: dict[str, dict[str, str]] = {
        # --- Regressão Linear / Múltipla ---
        "fit_intercept": {
            "explicacao_negocio": (
                "Define se o modelo pode calcular um valor-base inicial antes de considerar "
                "as características do imóvel. Quando ativado, o modelo reconhece que mesmo um "
                "imóvel com metragem zero teria um valor de mercado mínimo implícito."
            ),
            "impacto_aumentar": "N/A — parâmetro booleano. Ativado: permite preço-base; desativado: força a equação a passar pela origem.",
            "impacto_diminuir": "N/A — parâmetro booleano. Sem preço-base, qualquer previsão parte de zero.",
            "valor_true": "O modelo tem liberdade para definir um patamar mínimo de preço, refletindo custos de mercado independentes das características físicas.",
            "valor_false": "O modelo é forçado a prever R$ 0 quando todas as características são zero. Raramente é o comportamento ideal para imóveis.",
        },
        "positive": {
            "explicacao_negocio": (
                "Obriga o modelo a tratar todas as variáveis como tendo efeito positivo no preço. "
                "Por exemplo, impede o modelo de concluir que 'mais metragem reduz o preço'."
            ),
            "impacto_aumentar": "N/A — parâmetro booleano. Ativado: impede coeficientes negativos.",
            "impacto_diminuir": "N/A — parâmetro booleano. Desativado: o modelo pode encontrar relações negativas entre variáveis e preço.",
            "valor_true": "O modelo só atribuirá efeitos positivos às variáveis. Útil quando se sabe que mais metragem, quartos ou vagas sempre valoriza.",
            "valor_false": "O modelo tem liberdade total para aprender relações positivas e negativas entre variáveis e preço.",
        },
        # --- Polinomial ---
        "poly__degree": {
            "explicacao_negocio": (
                "Define o nível de complexidade das relações matemáticas que o modelo pode aprender. "
                "Com degree=2, o modelo pode representar que o impacto da metragem não é linear — "
                "um apartamento de 200m² pode não valer exatamente o dobro de um de 100m²."
            ),
            "impacto_aumentar": "O modelo consegue capturar relações mais sutis entre variáveis e preço, mas aumenta o risco de memorizar casos específicos do histórico.",
            "impacto_diminuir": "O modelo se aproxima de uma relação direta e linear entre cada variável e o preço.",
            "valor_2": "Grau 2 — o modelo pode representar curvas simples, como 'o valor por m² cresce menos rápido em imóveis muito grandes'.",
            "valor_3": "Grau 3 — relações mais complexas e potencialmente menos interpretáveis. Risco maior de overfitting.",
        },
        "linear__fit_intercept": {
            "explicacao_negocio": "Mesmo significado que fit_intercept: permite um valor-base na equação do modelo polinomial.",
            "impacto_aumentar": "N/A — booleano.",
            "impacto_diminuir": "N/A — booleano.",
            "valor_true": "Preço-base permitido no modelo polinomial.",
            "valor_false": "Sem preço-base — equação parte da origem.",
        },
        # --- Ridge / Lasso / Elastic Net ---
        "alpha": {
            "explicacao_negocio": (
                "Define a intensidade do 'freio' aplicado aos coeficientes do modelo. "
                "É como calibrar o quanto o modelo pode confiar em uma única variável para explicar o preço. "
                "Um alpha alto impede que o modelo 'aposte tudo' em uma característica específica, "
                "tornando as previsões mais equilibradas e robustas."
            ),
            "impacto_aumentar": (
                "O modelo fica mais conservador: distribui a importância entre mais variáveis, "
                "reduzindo o impacto de qualquer característica isolada. "
                "Risco: pode suavizar demais e perder precisão em mercados com forte influência de uma variável específica."
            ),
            "impacto_diminuir": (
                "O modelo se aproxima de uma regressão linear pura, sem freio. "
                "Pode aprender relações mais intensas de variáveis individuais, "
                "mas fica mais sensível a variações nos dados de treino."
            ),
            "valor_generico": (
                "Este valor equilibra a precisão do ajuste histórico com a robustez para novos imóveis. "
                "Foi escolhido automaticamente pelo Grid Search como o que minimiza o erro médio de previsão."
            ),
        },
        "l1_ratio": {
            "explicacao_negocio": (
                "Controla o equilíbrio entre dois tipos de regularização no Elastic Net. "
                "Valores próximos de 1 tornam o modelo mais parecido com o Lasso (pode zerar variáveis). "
                "Valores próximos de 0 o tornam mais parecido com o Ridge (distribui influência entre todas)."
            ),
            "impacto_aumentar": "O modelo tende a eliminar variáveis menos relevantes, ficando mais seletivo.",
            "impacto_diminuir": "O modelo mantém todas as variáveis ativas, distribuindo a influência entre elas.",
            "valor_generico": (
                "Este valor foi escolhido pelo Grid Search como o melhor equilíbrio entre "
                "seletividade de variáveis e distribuição de influência para este dataset imobiliário."
            ),
        },
        # --- Árvore de Decisão / Random Forest / Gradient Boosting ---
        "max_depth": {
            "explicacao_negocio": (
                "Define quantos níveis de perguntas o modelo pode encadear para chegar a uma previsão de preço. "
                "Exemplo: 'A metragem é maior que 150m²?' → 'A zona é Sul?' → 'Tem mais de 2 vagas?' "
                "Cada nível adicional permite regras mais específicas."
            ),
            "impacto_aumentar": (
                "O modelo pode criar regras mais detalhadas e específicas para subgrupos de imóveis, "
                "aumentando a precisão no histórico. Risco: pode memorizar padrões que não se repetem."
            ),
            "impacto_diminuir": (
                "O modelo produz regras mais gerais e aplicáveis a um conjunto maior de imóveis. "
                "Mais estável, porém pode perder precisão em casos muito específicos."
            ),
            "valor_none": "Sem limite de profundidade — o modelo cresce até que cada grupo terminal tenha amostras mínimas. Risco elevado de overfitting.",
            "valor_generico": (
                "Este número de níveis foi identificado pelo Grid Search como o equilíbrio ideal entre "
                "especificidade das regras e generalização para novos imóveis."
            ),
        },
        "min_samples_split": {
            "explicacao_negocio": (
                "Define o número mínimo de imóveis que um grupo precisa ter para que o modelo "
                "tente subdividi-lo em subgrupos mais específicos. "
                "Funciona como um critério mínimo de evidência antes de criar uma nova regra."
            ),
            "impacto_aumentar": "O modelo exige mais evidências antes de criar subdivisões, produzindo regras mais conservadoras e generalizáveis.",
            "impacto_diminuir": "O modelo pode criar regras com base em grupos muito pequenos, arriscando memorizar exceções.",
            "valor_generico": "Este valor define que grupos com menos imóveis que o limite não serão mais subdivididos.",
        },
        "min_samples_leaf": {
            "explicacao_negocio": (
                "Define o número mínimo de imóveis que devem existir numa conclusão final do modelo. "
                "É como exigir que cada previsão seja baseada em pelo menos N casos reais do histórico."
            ),
            "impacto_aumentar": "Cada previsão é baseada em mais casos históricos, tornando-a mais estável e menos específica.",
            "impacto_diminuir": "O modelo pode fazer previsões baseadas em poucos imóveis, aumentando sensibilidade a casos atípicos.",
            "valor_generico": "Este valor garante que toda previsão final seja suportada por pelo menos esse número de imóveis do histórico.",
        },
        "max_features": {
            "explicacao_negocio": (
                "Define quantas características (metragem, quartos, vagas, zona) o modelo considera "
                "em cada decisão de subdivisão. Reduzir este número força o modelo a diversificar "
                "quais variáveis usa, tornando-o menos dependente de uma única característica."
            ),
            "impacto_aumentar": "O modelo avalia mais características em cada decisão, potencialmente criando regras mais precisas.",
            "impacto_diminuir": "O modelo é forçado a trabalhar com um subconjunto de características, criando mais diversidade interna.",
            "valor_sqrt": "Raiz quadrada do total de variáveis — padrão recomendado para Random Forest, equilibra diversidade e precisão.",
            "valor_generico": "Este percentual das características é avaliado em cada etapa de decisão.",
        },
        # --- Random Forest / Gradient Boosting / XGBoost / LightGBM ---
        "n_estimators": {
            "explicacao_negocio": (
                "Define quantas árvores de decisão são construídas e combinadas para gerar a previsão final. "
                "É como consultar N especialistas e fazer uma média das opiniões. "
                "Mais árvores geralmente produz previsões mais estáveis."
            ),
            "impacto_aumentar": "Previsões mais estáveis e robustas, mas com maior custo de processamento a cada previsão.",
            "impacto_diminuir": "Processamento mais rápido, mas previsões potencialmente menos estáveis.",
            "valor_generico": "Este número de árvores foi definido pelo Grid Search como o equilíbrio entre estabilidade e eficiência.",
        },
        "learning_rate": {
            "explicacao_negocio": (
                "Controla o tamanho do passo que o modelo dá a cada etapa de correção. "
                "Um learning_rate pequeno significa que o modelo aprende de forma mais cautelosa, "
                "fazendo pequenas correções sucessivas. Um valor alto significa correções mais agressivas."
            ),
            "impacto_aumentar": "O modelo aprende mais rápido, mas pode ultrapassar o ponto ideal e oscilar nas previsões.",
            "impacto_diminuir": "O aprendizado é mais gradual e controlado. Normalmente requer mais árvores (n_estimators) para compensar.",
            "valor_generico": (
                "Este valor foi selecionado pelo Grid Search para equilibrar velocidade de aprendizado "
                "e estabilidade das previsões."
            ),
        },
        "subsample": {
            "explicacao_negocio": (
                "Define qual percentual dos imóveis do histórico é usado em cada etapa de treinamento. "
                "Usar menos que 100% introduz variabilidade saudável que torna o modelo mais robusto."
            ),
            "impacto_aumentar": "Usa mais dados por etapa, podendo melhorar a precisão em dados de treino.",
            "impacto_diminuir": "Usa menos dados por etapa, aumentando a diversidade interna e reduzindo risco de overfitting.",
            "valor_generico": f"Este percentual da base é amostrado a cada iteração.",
        },
        # --- XGBoost / LightGBM ---
        "colsample_bytree": {
            "explicacao_negocio": (
                "Define o percentual das variáveis (metragem, quartos, vagas, zona) "
                "que cada árvore pode usar. Semelhante ao max_features do Random Forest."
            ),
            "impacto_aumentar": "Cada árvore tem acesso a mais variáveis, potencialmente capturando mais relações.",
            "impacto_diminuir": "Cada árvore trabalha com menos variáveis, forçando diversidade no conjunto de árvores.",
            "valor_generico": "Este percentual de variáveis é disponibilizado para cada árvore individualmente.",
        },
        # --- LightGBM ---
        "num_leaves": {
            "explicacao_negocio": (
                "Define o número máximo de grupos (conclusões) que cada árvore pode criar. "
                "Mais folhas significa segmentações mais granulares do mercado imobiliário."
            ),
            "impacto_aumentar": "Árvores mais complexas com segmentações mais específicas. Risco de overfitting.",
            "impacto_diminuir": "Árvores mais simples com segmentações mais amplas e generalizáveis.",
            "valor_generico": "Este número de grupos finais por árvore foi escolhido pelo Grid Search.",
        },
        "min_child_samples": {
            "explicacao_negocio": (
                "Número mínimo de imóveis que devem existir em cada conclusão final da árvore. "
                "Equivalente ao min_samples_leaf, mas específico do LightGBM."
            ),
            "impacto_aumentar": "Exige mais evidências por conclusão, tornando o modelo mais conservador.",
            "impacto_diminuir": "Permite conclusões baseadas em poucos imóveis, aumentando o risco de memorização.",
            "valor_generico": "Cada conclusão deve ser suportada por pelo menos este número de imóveis.",
        },
        # --- CatBoost ---
        "iterations": {
            "explicacao_negocio": (
                "Número máximo de árvores que o CatBoost pode construir. "
                "Equivalente ao n_estimators de outros modelos boosting."
            ),
            "impacto_aumentar": "Mais ciclos de correção, potencialmente maior precisão, com maior custo computacional.",
            "impacto_diminuir": "Menos ciclos, treinamento mais rápido, mas possivelmente menos refinado.",
            "valor_generico": "Número máximo de iterações de boosting permitidas.",
        },
        "depth": {
            "explicacao_negocio": (
                "Profundidade máxima de cada árvore no CatBoost. "
                "Mesmo conceito do max_depth de outros modelos."
            ),
            "impacto_aumentar": "Árvores mais profundas capturam relações mais específicas.",
            "impacto_diminuir": "Árvores mais rasas produzem regras mais gerais.",
            "valor_generico": "Cada árvore pode ter no máximo este número de níveis de decisão.",
        },
        "l2_leaf_reg": {
            "explicacao_negocio": (
                "Regularização aplicada nos valores das folhas finais do CatBoost. "
                "Equivalente ao alpha do Ridge aplicado às conclusões das árvores."
            ),
            "impacto_aumentar": "Conclusões finais mais conservadoras, reduzindo previsões extremas.",
            "impacto_diminuir": "Conclusões menos controladas, podendo gerar previsões mais agressivas.",
            "valor_generico": "Este nível de regularização controla a intensidade das previsões finais de cada folha.",
        },
        "verbose": {
            "explicacao_negocio": "Controla se o modelo exibe mensagens de progresso durante o treinamento. Sem impacto na lógica de previsão.",
            "impacto_aumentar": "N/A — booleano de controle de logs.",
            "impacto_diminuir": "N/A — booleano de controle de logs.",
            "valor_false": "Treinamento silencioso — sem mensagens de progresso.",
            "valor_true": "Exibe progresso detalhado a cada iteração.",
        },
        # --- SVR ---
        "kernel": {
            "explicacao_negocio": (
                "Define o formato das relações que o SVR consegue aprender entre as variáveis e o preço. "
                "'rbf' (radial): capaz de aprender relações complexas e não-lineares. "
                "'linear': assume relações diretas e proporcionais."
            ),
            "impacto_aumentar": "N/A — parâmetro categórico.",
            "impacto_diminuir": "N/A — parâmetro categórico.",
            "valor_rbf": "O modelo pode aprender relações complexas e não-lineares entre variáveis e preço.",
            "valor_linear": "O modelo assume que a relação entre cada variável e o preço é direta e proporcional.",
        },
        "C": {
            "explicacao_negocio": (
                "Define o quanto o modelo SVR se preocupa em corrigir erros de previsão. "
                "Um C alto significa que o modelo tentará minimizar cada erro individualmente, "
                "podendo se tornar muito específico para os dados históricos."
            ),
            "impacto_aumentar": "O modelo tenta corrigir mais erros, ficando mais ajustado ao histórico. Risco de overfitting.",
            "impacto_diminuir": "O modelo aceita mais erros em troca de maior simplicidade e robustez.",
            "valor_generico": "Este valor equilibra a tolerância a erros com a robustez das previsões.",
        },
        "epsilon": {
            "explicacao_negocio": (
                "Define a faixa de erro que o modelo SVR considera aceitável sem penalização. "
                "Erros dentro desta faixa são ignorados durante o treinamento. "
                "Em termos práticos, é como definir uma margem de tolerância interna."
            ),
            "impacto_aumentar": "O modelo tolera erros maiores, produzindo previsões mais conservadoras.",
            "impacto_diminuir": "O modelo exige alta precisão, podendo ficar mais sensível a variações nos dados.",
            "valor_generico": "Previsões dentro desta margem (em escala normalizada) não são penalizadas durante o treinamento.",
        },
        "gamma": {
            "explicacao_negocio": (
                "Define o alcance da influência de cada imóvel do histórico sobre as previsões. "
                "Com 'scale' ou 'auto', este valor é ajustado automaticamente. "
                "Um gamma alto significa que apenas imóveis muito similares influenciam a previsão."
            ),
            "impacto_aumentar": "Influência mais localizada — apenas imóveis muito próximos ao avaliado impactam a previsão.",
            "impacto_diminuir": "Influência mais ampla — imóveis com características mais diversas podem impactar a previsão.",
            "valor_scale": "Ajustado automaticamente pela variação dos dados — recomendado como ponto de partida.",
            "valor_auto": "Calculado como 1 / número de variáveis.",
        },
        # --- Rede Neural ---
        "hidden_layer_sizes": {
            "explicacao_negocio": (
                "Define a estrutura interna da rede neural: quantas camadas de processamento existem "
                "e quantos neurônios cada camada tem. Uma rede maior pode capturar relações mais complexas, "
                "mas exige mais dados e mais tempo de treinamento."
            ),
            "impacto_aumentar": "Rede mais profunda ou larga, com maior capacidade de aprender relações complexas entre variáveis e preço.",
            "impacto_diminuir": "Rede mais simples, mais rápida de treinar, com menor risco de overfitting.",
            "valor_64": "Rede simples com 64 neurônios em uma camada — adequada para relações relativamente diretas.",
            "valor_128": "Rede moderada com 128 neurônios — maior capacidade de aprendizado.",
            "valor_64_32": "Duas camadas (64 e 32 neurônios) — estrutura em funil, progressivamente mais específica.",
            "valor_128_64": "Duas camadas maiores (128 e 64 neurônios) — maior capacidade de representação.",
            "valor_generico": "Esta configuração de camadas e neurônios foi selecionada pelo Grid Search.",
        },
        "learning_rate_init": {
            "explicacao_negocio": (
                "Define a velocidade inicial com que a rede neural ajusta seus pesos durante o treinamento. "
                "É análogo ao learning_rate dos modelos boosting, mas aplicado à rede neural."
            ),
            "impacto_aumentar": "Ajustes iniciais mais agressivos, podendo convergir mais rápido ou oscilar.",
            "impacto_diminuir": "Ajustes mais cautelosos, aprendizado mais gradual e estável.",
            "valor_generico": "Taxa de aprendizado inicial selecionada pelo Grid Search para esta rede neural.",
        },
    }

    # Mapeamento de TipoModelo.value → nome amigável para o relatório
    _NOMES_AMIGAVEIS: dict[str, str] = {
        "regressao_linear": "Regressão Linear",
        "regressao_multipla": "Regressão Linear Múltipla",
        "regressao_polinomial": "Regressão Polinomial",
        "ridge": "Ridge (Regressão com Regularização L2)",
        "lasso": "Lasso (Regressão com Regularização L1)",
        "elastic_net": "Elastic Net (Regularização L1 + L2)",
        "arvore_decisao": "Árvore de Decisão",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "svr": "SVR — Máquina de Vetores de Suporte para Regressão",
        "rede_neural": "Rede Neural (MLPRegressor)",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
    }

    def _limpar_nome_param(self, chave_pipeline: str) -> str:
        """Remove prefixo 'modelo__' e sub-prefixos de etapas da pipeline."""
        nome = chave_pipeline
        for prefixo in ("modelo__poly__", "modelo__linear__", "modelo__"):
            if nome.startswith(prefixo):
                nome = nome[len(prefixo):]
                break
        return nome

    def _formatar_valor(self, valor: str | int | float | bool | tuple[int, ...] | None) -> str:
        """Formata o valor do parâmetro para exibição legível."""
        if valor is None:
            return "automático (sem limite)"
        if isinstance(valor, bool):
            return "Sim" if valor else "Não"
        if isinstance(valor, float):
            return f"{valor:.4g}"
        if isinstance(valor, tuple):
            return " → ".join(str(n) for n in valor) + f" neurônios ({len(valor)} camada{'s' if len(valor) > 1 else ''})"
        return str(valor)

    def _obter_interpretacao_valor(
        self,
        param_nome: str,
        valor: str | int | float | bool | tuple[int, ...] | None,
        explicacoes: dict[str, str],
    ) -> str:
        """Retorna a interpretação específica do valor selecionado, quando disponível."""
        if isinstance(valor, bool):
            chave_val = "valor_true" if valor else "valor_false"
            return explicacoes.get(chave_val, explicacoes.get("valor_generico", ""))
        if isinstance(valor, str):
            chave_val = f"valor_{valor}"
            return explicacoes.get(chave_val, explicacoes.get("valor_generico", ""))
        if valor is None:
            return explicacoes.get("valor_none", explicacoes.get("valor_generico", ""))
        if isinstance(valor, tuple):
            chave_val = "valor_" + "_".join(str(n) for n in valor)
            return explicacoes.get(chave_val, explicacoes.get("valor_generico", ""))
        return explicacoes.get("valor_generico", "")

    def gerar_relatorio_negocio(
        self,
        nome_modelo: str,
        melhores_parametros: dict[str, str | int | float | bool | tuple[int, ...] | None],
    ) -> tuple[list[dict[str, str]], str]:
        """Gera explicação de negócio para os hiperparâmetros selecionados pelo Grid Search.

        Parameters
        ----------
        nome_modelo : str
            Valor do TipoModelo (ex: 'random_forest', 'ridge').
        melhores_parametros : dict
            Dicionário de melhores parâmetros retornado pelo GridSearchCV (com prefixo 'modelo__').

        Returns
        -------
        tuple[list[dict[str, str]], str]
            Lista de dicionários estruturados por parâmetro e relatório em Markdown.
        """
        nome_amigavel = self._NOMES_AMIGAVEIS.get(nome_modelo, nome_modelo.replace("_", " ").title())
        explicacoes_estruturadas: list[dict[str, str]] = []

        for chave_pipeline, valor in melhores_parametros.items():
            param_nome = self._limpar_nome_param(chave_pipeline)
            valor_fmt = self._formatar_valor(valor)  # type: ignore[arg-type]
            explicacoes = self._CATALOGO.get(param_nome, {})

            exp_negocio = explicacoes.get("explicacao_negocio", f"Parâmetro técnico do modelo {nome_amigavel}.")
            impacto_aumentar = explicacoes.get("impacto_aumentar", "Consulte a documentação técnica.")
            impacto_diminuir = explicacoes.get("impacto_diminuir", "Consulte a documentação técnica.")
            interpretacao_valor = self._obter_interpretacao_valor(param_nome, valor, explicacoes)  # type: ignore[arg-type]

            explicacoes_estruturadas.append(
                {
                    "modelo": nome_amigavel,
                    "parametro": param_nome,
                    "valor_selecionado": valor_fmt,
                    "explicacao_negocio": exp_negocio,
                    "impacto_se_aumentar": impacto_aumentar,
                    "impacto_se_diminuir": impacto_diminuir,
                    "interpretacao_do_valor": interpretacao_valor,
                }
            )

        md = self._gerar_markdown(nome_amigavel, explicacoes_estruturadas)
        return explicacoes_estruturadas, md

    def _gerar_markdown(
        self,
        nome_amigavel: str,
        params: list[dict[str, str]],
    ) -> str:
        """Gera relatório Markdown formatado para apresentação à equipe de negócio."""
        if not params:
            return ""

        linhas: list[str] = [
            f"# Hiperparâmetros Selecionados — {nome_amigavel}",
            "",
            "> **Para a Equipe de Negócio**  ",
            "> Os valores abaixo foram escolhidos automaticamente pelo processo de otimização (Grid Search)  ",
            "> para maximizar a precisão das previsões de preço de imóveis.",
            "",
            "---",
            "",
            "## Resumo dos Parâmetros Selecionados",
            "",
            "| Parâmetro | Valor Escolhido | Significado Resumido |",
            "| :--- | :---: | :--- |",
        ]

        for p in params:
            resumo = p["explicacao_negocio"][:80].rstrip() + ("..." if len(p["explicacao_negocio"]) > 80 else "")
            linhas.append(f"| `{p['parametro']}` | **{p['valor_selecionado']}** | {resumo} |")

        linhas += ["", "---", "", "## Explicação Detalhada de Cada Parâmetro", ""]

        for p in params:
            linhas += [
                f"### `{p['parametro']}` = {p['valor_selecionado']}",
                "",
                f"**O que controla**: {p['explicacao_negocio']}",
                "",
            ]
            if p["interpretacao_do_valor"]:
                linhas += [
                    f"**Com este valor**: {p['interpretacao_do_valor']}",
                    "",
                ]
            linhas += [
                f"**Se este valor fosse maior**: {p['impacto_se_aumentar']}",
                "",
                f"**Se este valor fosse menor**: {p['impacto_se_diminuir']}",
                "",
                "---",
                "",
            ]

        linhas += [
            "## 💡 Mensagem para o Negócio",
            "",
            f"O modelo **{nome_amigavel}** foi configurado com {len(params)} hiperparâmetro(s) "
            f"otimizado(s) automaticamente. Esses valores representam o melhor equilíbrio encontrado "
            f"entre precisão no histórico de imóveis e capacidade de generalizar para novos casos.",
            "",
            "*Relatório gerado automaticamente após o treinamento.*",
        ]

        return "\n".join(linhas)
