"""Extrator de coeficientes e equações matemáticas para modelos lineares."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class ExtratorCoeficientes:
    """Extrai coeficientes, intercepto e equações analíticas de pipelines lineares."""

    def extrair(
        self,
        pipeline_ajustada: Pipeline,
    ) -> tuple[pd.DataFrame, float, str]:
        """Extrai a tabela de coeficientes, o intercepto e a equação formatada.

        Parameters
        ----------
        pipeline_ajustada : Pipeline
            Pipeline treinada contendo pré-processador e estimador linear.

        Returns
        -------
        tuple[pd.DataFrame, float, str]
            Tabela de coeficientes, intercepto numérico e equação em texto puro.
        """
        preprocessador = pipeline_ajustada.named_steps.get("pre") or pipeline_ajustada.steps[0][1]
        estimador = pipeline_ajustada.named_steps.get("modelo") or pipeline_ajustada.steps[-1][1]

        # Se for pipeline aninhada (ex: regressão polinomial)
        if isinstance(estimador, Pipeline):
            estimador_final = estimador.named_steps.get("linear") or estimador.steps[-1][1]
        else:
            estimador_final = estimador

        if not hasattr(estimador_final, "coef_"):
            raise ValueError("O estimador fornecido não possui o atributo 'coef_'.")

        coeficientes: np.ndarray = np.ravel(estimador_final.coef_)
        intercepto: float = float(getattr(estimador_final, "intercept_", 0.0))

        # Extrai nomes das features transformadas
        nomes_features: list[str] = []
        if hasattr(preprocessador, "get_feature_names_out"):
            nomes_features = [str(n) for n in preprocessador.get_feature_names_out()]
        else:
            nomes_features = [f"x{i}" for i in range(len(coeficientes))]

        # Ajuste de tamanho caso regressão polinomial tenha expandido as features
        if len(nomes_features) != len(coeficientes):
            if hasattr(estimador, "named_steps") and "poly" in estimador.named_steps:
                poly = estimador.named_steps["poly"]
                if hasattr(poly, "get_feature_names_out"):
                    nomes_features = [str(n) for n in poly.get_feature_names_out(nomes_features)]
                else:
                    nomes_features = [f"poly_{i}" for i in range(len(coeficientes))]
            else:
                nomes_features = [f"feature_{i}" for i in range(len(coeficientes))]

        registros = []
        termos_equacao = [f"{intercepto:,.2f}"]

        for nome, coef in zip(nomes_features, coeficientes):
            coef_val = float(coef)
            sinal = "+" if coef_val >= 0 else "-"
            magnitude = float(abs(coef_val))
            registros.append(
                {
                    "feature": nome,
                    "coeficiente": coef_val,
                    "sinal": sinal,
                    "magnitude": magnitude,
                }
            )
            termos_equacao.append(f"({sinal} {magnitude:,.2f} * {nome})")

        df_coef = (
            pd.DataFrame(registros)
            .sort_values(by="magnitude", ascending=False)
            .reset_index(drop=True)
        )

        equacao = "Valor_da_Venda = " + " + ".join(termos_equacao)
        return df_coef, intercepto, equacao

    def gerar_equacao_txt(
        self,
        pipeline_ajustada: Pipeline,
        nome_modelo: str = "",
    ) -> str:
        """Gera o documento textual (.txt) contendo a equação matemática completa do modelo.

        Suporta modelos lineares, polinomiais, árvores de decisão, florestas aleatórias,
        modelos de boosting, SVR, redes neurais e ensembles.

        Parameters
        ----------
        pipeline_ajustada : Pipeline
            Pipeline treinada contendo pré-processador e modelo estimador.
        nome_modelo : str
            Identificador ou nome amigável do modelo.

        Returns
        -------
        str
            Conteúdo textual formatado para o artefato .txt no MLflow.
        """
        preprocessador = pipeline_ajustada.named_steps.get("pre") or pipeline_ajustada.steps[0][1]
        estimador = pipeline_ajustada.named_steps.get("modelo") or pipeline_ajustada.steps[-1][1]

        # Se for pipeline aninhada (ex: regressão polinomial)
        if isinstance(estimador, Pipeline):
            estimador_final = estimador.named_steps.get("linear") or estimador.steps[-1][1]
        else:
            estimador_final = estimador

        nome_exibicao = nome_modelo or estimador_final.__class__.__name__

        # 1. Modelos Lineares e Regularizados (possuem coef_)
        if hasattr(estimador_final, "coef_"):
            return self._gerar_txt_modelo_linear(
                pipeline_ajustada=pipeline_ajustada,
                estimador_final=estimador_final,
                nome_modelo=nome_exibicao,
            )

        # Extrai nomes das features transformadas para modelos não lineares
        nomes_features: list[str] = []
        if hasattr(preprocessador, "get_feature_names_out"):
            try:
                nomes_features = [str(n) for n in preprocessador.get_feature_names_out()]
            except Exception:
                nomes_features = []

        classe_nome = estimador_final.__class__.__name__

        # 2. Árvore de Decisão
        if "DecisionTree" in classe_nome or "Arvore" in classe_nome:
            return self._gerar_txt_arvore_decisao(
                estimador_final=estimador_final,
                nomes_features=nomes_features,
                nome_modelo=nome_exibicao,
            )

        # 3. Random Forest e Bagging
        if "RandomForest" in classe_nome or "Bagging" in classe_nome or "Floresta" in classe_nome:
            return self._gerar_txt_random_forest(
                estimador_final=estimador_final,
                nomes_features=nomes_features,
                nome_modelo=nome_exibicao,
            )

        # 4. Modelos de Boosting (GradientBoosting, XGBoost, LightGBM, CatBoost)
        if any(b in classe_nome for b in ("GradientBoosting", "XGB", "LGBM", "CatBoost")):
            return self._gerar_txt_boosting(
                estimador_final=estimador_final,
                nomes_features=nomes_features,
                nome_modelo=nome_exibicao,
            )

        # 5. SVR (Support Vector Regression)
        if "SVR" in classe_nome:
            return self._gerar_txt_svr(
                estimador_final=estimador_final,
                nome_modelo=nome_exibicao,
            )

        # 6. Rede Neural (MLPRegressor)
        if "MLP" in classe_nome:
            return self._gerar_txt_mlp(
                estimador_final=estimador_final,
                nome_modelo=nome_exibicao,
            )

        # 7. Modelos de Votação e Stacking (Ensembles)
        if hasattr(estimador_final, "estimators_"):
            return self._gerar_txt_ensemble(
                estimador_final=estimador_final,
                nome_modelo=nome_exibicao,
            )

        # Fallback genérico para outros estimadores
        return self._gerar_txt_generico(estimador_final, nome_exibicao)

    def _gerar_txt_modelo_linear(
        self,
        pipeline_ajustada: Pipeline,
        estimador_final: object,
        nome_modelo: str,
    ) -> str:
        df_coef, intercepto, equacao = self.extrair(pipeline_ajustada)
        linhas: list[str] = [
            "=" * 80,
            f"EQUAÇÃO DO MODELO: {nome_modelo.upper()}",
            "=" * 80,
            "",
            "1. FORMULAÇÃO MATEMÁTICA GERAL:",
            "   Valor_da_Venda = β₀ + ∑_{i=1}^{p} (βᵢ * Xᵢ)",
            "",
            "2. EQUAÇÃO ANALÍTICA ESTIMADA:",
            f"   {equacao}",
            "",
            f"3. INTERCEPTO BASE (β₀): R$ {intercepto:,.2f}",
            "",
            "4. COEFICIENTES DAS VARIÁVEIS (ordenados por magnitude de impacto):",
            f"   {'Variável / Feature':<40} {'Coeficiente (R$)':<20} {'Impacto':<10}",
            "   " + "-" * 70,
        ]
        for _, row in df_coef.iterrows():
            feat = str(row["feature"])[:38]
            coef = float(row["coeficiente"])
            sinal = str(row["sinal"])
            tipo_imp = "Positivo" if sinal == "+" else "Negativo"
            linhas.append(f"   {feat:<40} R$ {coef:>15,.2f}   [{tipo_imp}]")

        linhas.extend(
            [
                "",
                "=" * 80,
                "Artefato gerado automaticamente pelo pipeline de Machine Learning Imobiliário.",
                "=" * 80,
            ]
        )
        return "\n".join(linhas)

    def _gerar_txt_arvore_decisao(
        self,
        estimador_final: object,
        nomes_features: list[str],
        nome_modelo: str,
    ) -> str:
        from sklearn.tree import export_text

        n_folhas = getattr(estimador_final, "get_n_leaves", lambda: 0)()
        profundidade = getattr(estimador_final, "get_depth", lambda: 0)()

        regras_arvore: str = ""
        try:
            fn = nomes_features if len(nomes_features) == getattr(estimador_final, "n_features_in_", len(nomes_features)) else None
            regras_arvore = export_text(estimador_final, feature_names=fn, max_depth=5)
        except Exception:
            regras_arvore = "Árvore de decisão com partições recursivas no espaço de atributos."

        linhas: list[str] = [
            "=" * 80,
            f"EQUAÇÃO DO MODELO: {nome_modelo.upper()} (Árvore de Decisão)",
            "=" * 80,
            "",
            "1. FORMULAÇÃO MATEMÁTICA ANALÍTICA:",
            "   Valor_da_Venda = ∑_{m=1}^{M} c_m * 𝕀(X ∈ R_m)",
            "",
            "   Onde:",
            f"   - M = {n_folhas} regiões particionadas ortogonalmente (nós folha).",
            "   - c_m = valor predito constante em cada folha m (preço médio histórico da região).",
            "   - 𝕀(X ∈ R_m) = função indicadora: 1 se as características do imóvel satisfazem",
            "     as condições de decisão da partição R_m, e 0 caso contrário.",
            f"   - Profundidade máxima da árvore: {profundidade} níveis de decisão encadeada.",
            "",
            "2. ESTRUTURA PRINCIPAL DE REGRAS DE DECISÃO (Árvore Decisória):",
            regras_arvore,
            "",
            "=" * 80,
            "Artefato gerado automaticamente pelo pipeline de Machine Learning Imobiliário.",
            "=" * 80,
        ]
        return "\n".join(linhas)

    def _gerar_txt_random_forest(
        self,
        estimador_final: object,
        nomes_features: list[str],
        nome_modelo: str,
    ) -> str:
        n_estimators = getattr(estimador_final, "n_estimators", 100)
        max_depth = getattr(estimador_final, "max_depth", "Ilimitado")
        max_features = getattr(estimador_final, "max_features", "auto")

        linhas: list[str] = [
            "=" * 80,
            f"EQUAÇÃO DO MODELO: {nome_modelo.upper()} (Floresta Aleatória / Random Forest)",
            "=" * 80,
            "",
            "1. FORMULAÇÃO MATEMÁTICA DE AGREGAÇÃO (Bagging Ensemble):",
            "   Valor_da_Venda = (1 / B) * ∑_{b=1}^{B} T_b(X)",
            "",
            "   Onde:",
            f"   - B = {n_estimators} árvores de decisão independentes treinadas via bootstrap.",
            "   - T_b(X) = predição da b-ésima árvore individual para o vetor de atributos X.",
            "   - Cada árvore T_b(X) avalia um subespaço aleatório de variáveis e amostras.",
            "",
            "2. PARÂMETROS DA ARQUITETURA:",
            f"   - Número de árvores (n_estimators): {n_estimators}",
            f"   - Profundidade máxima (max_depth): {max_depth}",
            f"   - Features avaliadas por divisão (max_features): {max_features}",
            "",
        ]

        # Top importâncias se disponível
        importancias = getattr(estimador_final, "feature_importances_", None)
        if importancias is not None and len(importancias) > 0 and len(nomes_features) == len(importancias):
            ranking = sorted(zip(nomes_features, importancias), key=lambda x: x[1], reverse=True)
            linhas.extend(
                [
                    "3. IMPORTÂNCIA RELATIVA DAS VARIÁVEIS NA PREDIÇÃO (Gini / MSE Reduction):",
                    f"   {'Variável':<40} {'Peso Relativo (%)':<20}",
                    "   " + "-" * 60,
                ]
            )
            for feat, imp in ranking[:15]:
                linhas.append(f"   {feat[:38]:<40} {imp * 100.0:>16.2f}%")

        linhas.extend(
            [
                "",
                "=" * 80,
                "Artefato gerado automaticamente pelo pipeline de Machine Learning Imobiliário.",
                "=" * 80,
            ]
        )
        return "\n".join(linhas)

    def _gerar_txt_boosting(
        self,
        estimador_final: object,
        nomes_features: list[str],
        nome_modelo: str,
    ) -> str:
        learning_rate = getattr(estimador_final, "learning_rate", 0.1)
        n_estimators = getattr(estimador_final, "n_estimators", getattr(estimador_final, "iterations", 100))
        max_depth = getattr(estimador_final, "max_depth", getattr(estimador_final, "depth", "Padrão"))

        linhas: list[str] = [
            "=" * 80,
            f"EQUAÇÃO DO MODELO: {nome_modelo.upper()} (Boosting Aditivo por Gradiente)",
            "=" * 80,
            "",
            "1. FORMULAÇÃO MATEMÁTICA DE EXPANSÃO ADITIVA:",
            "   Valor_da_Venda = F₀(X) + η * ∑_{m=1}^{M} h_m(X)",
            "",
            "   Onde:",
            "   - F₀(X) = Predição base inicial (média/baseline dos preços).",
            f"   - η (eta) = Taxa de aprendizado (learning_rate = {learning_rate}).",
            f"   - M = {n_estimators} estágios de árvores de decisão aditivas.",
            "   - h_m(X) = m-ésima árvore ajustada aos pseudo-resíduos do estágio anterior.",
            "",
            "2. PARÂMETROS DA ARQUITETURA:",
            f"   - Taxa de aprendizado (learning_rate): {learning_rate}",
            f"   - Número de estágios / estimadores: {n_estimators}",
            f"   - Profundidade máxima das árvores: {max_depth}",
            "",
        ]

        importancias = getattr(estimador_final, "feature_importances_", None)
        if importancias is not None and len(importancias) > 0 and len(nomes_features) == len(importancias):
            ranking = sorted(zip(nomes_features, importancias), key=lambda x: x[1], reverse=True)
            linhas.extend(
                [
                    "3. IMPORTÂNCIA DAS VARIÁVEIS NA EXPANSÃO ADITIVA:",
                    f"   {'Variável':<40} {'Ganho Relativo (%)':<20}",
                    "   " + "-" * 60,
                ]
            )
            for feat, imp in ranking[:15]:
                linhas.append(f"   {feat[:38]:<40} {float(imp) * 100.0:>16.2f}%")

        linhas.extend(
            [
                "",
                "=" * 80,
                "Artefato gerado automaticamente pelo pipeline de Machine Learning Imobiliário.",
                "=" * 80,
            ]
        )
        return "\n".join(linhas)

    def _gerar_txt_svr(
        self,
        estimador_final: object,
        nome_modelo: str,
    ) -> str:
        kernel = getattr(estimador_final, "kernel", "rbf")
        c_param = getattr(estimador_final, "C", 1.0)
        epsilon = getattr(estimador_final, "epsilon", 0.1)
        gamma = getattr(estimador_final, "gamma", "scale")
        intercepto = float(getattr(estimador_final, "intercept_", [0.0])[0]) if hasattr(estimador_final, "intercept_") else 0.0
        n_suporte = len(getattr(estimador_final, "support_", []))

        linhas: list[str] = [
            "=" * 80,
            f"EQUAÇÃO DO MODELO: {nome_modelo.upper()} (Support Vector Regression - SVR)",
            "=" * 80,
            "",
            "1. FORMULAÇÃO MATEMÁTICA DUAL COM KERNEL:",
            "   Valor_da_Venda = b + ∑_{i ∈ SV} (αᵢ - αᵢ*) * K(Xᵢ, X)",
            "",
            "   Onde:",
            f"   - Kernel K(Xᵢ, X): {kernel.upper()} [exp(-γ * ||X - Xᵢ||²)]",
            f"   - Intercepto analítico (b): R$ {intercepto:,.2f}",
            f"   - Número de Vetores de Suporte (SV): {n_suporte}",
            f"   - Parâmetro de Regularização (C): {c_param}",
            f"   - Margem de Tolerância Invariante (ε): {epsilon}",
            f"   - Parâmetro Gama do Kernel (γ): {gamma}",
            "",
            "=" * 80,
            "Artefato gerado automaticamente pelo pipeline de Machine Learning Imobiliário.",
            "=" * 80,
        ]
        return "\n".join(linhas)

    def _gerar_txt_mlp(
        self,
        estimador_final: object,
        nome_modelo: str,
    ) -> str:
        camadas = getattr(estimador_final, "hidden_layer_sizes", (100,))
        ativacao = getattr(estimador_final, "activation", "relu")
        alpha = getattr(estimador_final, "alpha", 0.0001)
        learning_rate_init = getattr(estimador_final, "learning_rate_init", 0.001)

        linhas: list[str] = [
            "=" * 80,
            f"EQUAÇÃO DO MODELO: {nome_modelo.upper()} (Rede Neural Perceptron Multicamadas - MLP)",
            "=" * 80,
            "",
            "1. FORMULAÇÃO MATEMÁTICA MATRICIAL EM CAMADAS:",
            "   h⁽¹⁾ = g(W⁽¹⁾ * X + b⁽¹⁾)",
            "   h⁽ˡ⁾ = g(W⁽ˡ⁾ * h⁽ˡ⁻¹⁾ + b⁽ˡ⁾)   para l = 2, ..., L-1",
            "   Valor_da_Venda = W⁽ᵒᵘᵗ⁾ * h⁽ᴸ⁻¹⁾ + b⁽ᵒᵘᵗ⁾",
            "",
            "   Onde:",
            f"   - Função de Ativação g(z): {ativacao.upper()} [ex: max(0, z)]",
            f"   - Arquitetura de Camadas Ocultas: {camadas}",
            f"   - Fator de Regularização L2 (alpha): {alpha}",
            f"   - Taxa de Aprendizado Inicial: {learning_rate_init}",
            "",
            "=" * 80,
            "Artefato gerado automaticamente pelo pipeline de Machine Learning Imobiliário.",
            "=" * 80,
        ]
        return "\n".join(linhas)

    def _gerar_txt_ensemble(
        self,
        estimador_final: object,
        nome_modelo: str,
    ) -> str:
        linhas: list[str] = [
            "=" * 80,
            f"EQUAÇÃO DO MODELO: {nome_modelo.upper()} (Ensemble de Modelos)",
            "=" * 80,
            "",
            "1. FORMULAÇÃO MATEMÁTICA DE COMBINAÇÃO:",
            "   Valor_da_Venda = ∑_{k=1}^{K} wₖ * Pred_Modelo_k(X)",
            "",
            "   Onde:",
            "   - wₖ = peso ponderado de cada modelo componente no comitê preditivo.",
            "   - Pred_Modelo_k(X) = estimativa de preço fornecida pelo modelo k.",
            "",
            "=" * 80,
            "Artefato gerado automaticamente pelo pipeline de Machine Learning Imobiliário.",
            "=" * 80,
        ]
        return "\n".join(linhas)

    def _gerar_txt_generico(self, estimador_final: object, nome_modelo: str) -> str:
        linhas: list[str] = [
            "=" * 80,
            f"EQUAÇÃO DO MODELO: {nome_modelo.upper()}",
            "=" * 80,
            "",
            "1. FORMULAÇÃO MATEMÁTICA:",
            "   Valor_da_Venda = f(X; θ)",
            "",
            f"   Estimador: {estimador_final.__class__.__name__}",
            f"   Parâmetros: {getattr(estimador_final, 'get_params', lambda: {})()}",
            "",
            "=" * 80,
            "Artefato gerado automaticamente pelo pipeline de Machine Learning Imobiliário.",
            "=" * 80,
        ]
        return "\n".join(linhas)
