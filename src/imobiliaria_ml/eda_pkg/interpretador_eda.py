"""Módulo especializado na interpretação e análise aprofundada de EDA por Zona imobiliária."""

import numpy as np
import pandas as pd


class InterpretadorEda:
    """Interpreta padrões espaciais, dispersões de valor e métricas imobiliárias agrupadas por Zona."""

    def analisar_estatisticas_por_zona(
        self,
        dados: pd.DataFrame,
        coluna_zona: str = "Zona",
        coluna_alvo: str = "Valor_da_Venda",
        coluna_metragem: str = "Metragem",
    ) -> pd.DataFrame:
        """Calcula estatísticas descritivas completas agregadas por Zona geográfica.

        Parameters
        ----------
        dados : pd.DataFrame
            Conjunto de dados imobiliários.
        coluna_zona : str
            Coluna categórica indicativa da região.
        coluna_alvo : str
            Coluna contínua de preço de venda.
        coluna_metragem : str
            Coluna com a área construída do imóvel.

        Returns
        -------
        pd.DataFrame
            Tabela estruturada com contagens, médias, medianas, dispersões e R$/m² por Zona.
        """
        if coluna_zona not in dados.columns or coluna_alvo not in dados.columns:
            return pd.DataFrame()

        df_trabalho = dados.copy()
        total_amostras = len(df_trabalho)

        # Cálculo do valor por metro quadrado quando metragem disponível
        tem_metragem = coluna_metragem in df_trabalho.columns
        if tem_metragem:
            metragens_seguras = np.maximum(df_trabalho[coluna_metragem].values, 1.0)
            df_trabalho["Valor_m2"] = df_trabalho[coluna_alvo].values / metragens_seguras

        linhas_zonas: list[dict[str, float | int | str]] = []

        for zona, grupo in df_trabalho.groupby(coluna_zona, observed=False):
            nome_zona = str(zona)
            precos = grupo[coluna_alvo].dropna().values
            qtd = len(precos)
            if qtd == 0:
                continue

            pct_base = round((qtd / total_amostras) * 100.0, 2)
            preco_medio = float(np.mean(precos))
            preco_mediano = float(np.median(precos))
            preco_std = float(np.std(precos, ddof=1)) if qtd > 1 else 0.0
            cv_preco = round((preco_std / preco_medio) * 100.0, 2) if preco_medio > 0 else 0.0

            # Detecção de outliers de preço específicos da Zona via IQR
            q25 = float(np.percentile(precos, 25))
            q75 = float(np.percentile(precos, 75))
            iqr = q75 - q25
            lim_inf = q25 - 1.5 * iqr
            lim_sup = q75 + 1.5 * iqr
            outliers_qtd = int(np.sum((precos < lim_inf) | (precos > lim_sup)))
            outliers_pct = round((outliers_qtd / qtd) * 100.0, 2)

            linha: dict[str, float | int | str] = {
                "zona": nome_zona,
                "quantidade_imoveis": qtd,
                "representatividade_pct": pct_base,
                "preco_medio": round(preco_medio, 2),
                "preco_mediano": round(preco_mediano, 2),
                "desvio_padrao_preco": round(preco_std, 2),
                "coeficiente_variacao_pct": cv_preco,
                "preco_minimo": round(float(np.min(precos)), 2),
                "preco_maximo": round(float(np.max(precos)), 2),
                "outliers_zona_qtd": outliers_qtd,
                "outliers_zona_pct": outliers_pct,
            }

            if tem_metragem and "Valor_m2" in grupo.columns:
                m2_valores = grupo["Valor_m2"].dropna().values
                linha["preco_m2_medio"] = round(float(np.mean(m2_valores)), 2)
                linha["preco_m2_mediano"] = round(float(np.median(m2_valores)), 2)
                linha["metragem_media"] = round(float(np.mean(grupo[coluna_metragem].dropna())), 1)
                linha["metragem_mediana"] = round(float(np.median(grupo[coluna_metragem].dropna())), 1)

            if "Quartos" in grupo.columns:
                linha["quartos_medio"] = round(float(np.mean(grupo["Quartos"].dropna())), 1)
            if "Banheiros" in grupo.columns:
                linha["banheiros_medio"] = round(float(np.mean(grupo["Banheiros"].dropna())), 1)
            if "Vagas" in grupo.columns:
                linha["vagas_medio"] = round(float(np.mean(grupo["Vagas"].dropna())), 1)

            linhas_zonas.append(linha)

        if not linhas_zonas:
            return pd.DataFrame()

        df_resultado = pd.DataFrame(linhas_zonas)
        # Ordena prioritariamente por preço médio por m² ou preço mediano decrescente
        criterio_ordenacao = "preco_m2_medio" if "preco_m2_medio" in df_resultado.columns else "preco_mediano"
        return df_resultado.sort_values(by=criterio_ordenacao, ascending=False).reset_index(drop=True)

    def gerar_diagnostico_zonas(self, tabela_zonas: pd.DataFrame) -> dict[str, str]:
        """Extrai diagnósticos interpretativos de negócio a partir das métricas por Zona.

        Parameters
        ----------
        tabela_zonas : pd.DataFrame
            DataFrame estruturado gerado por analisar_estatisticas_por_zona.

        Returns
        -------
        dict[str, str]
            Dicionário com conclusões e insights de negócio sintetizados.
        """
        if tabela_zonas.empty or "zona" not in tabela_zonas.columns:
            return {}

        col_preco = "preco_mediano" if "preco_mediano" in tabela_zonas.columns else "preco_medio"
        col_m2 = "preco_m2_medio" if "preco_m2_medio" in tabela_zonas.columns else col_preco

        idx_mais_cara = int(tabela_zonas[col_m2].idxmax())
        idx_mais_barata = int(tabela_zonas[col_m2].idxmin())
        idx_maior_vol = int(tabela_zonas["quantidade_imoveis"].idxmax())
        idx_menor_vol = int(tabela_zonas["quantidade_imoveis"].idxmin())
        idx_maior_disp = int(tabela_zonas["coeficiente_variacao_pct"].idxmax())

        zona_top = str(tabela_zonas.loc[idx_mais_cara, "zona"])
        zona_base = str(tabela_zonas.loc[idx_mais_barata, "zona"])
        zona_vol = str(tabela_zonas.loc[idx_maior_vol, "zona"])
        zona_escassa = str(tabela_zonas.loc[idx_menor_vol, "zona"])
        zona_disp = str(tabela_zonas.loc[idx_maior_disp, "zona"])

        p_m2_top = float(tabela_zonas.loc[idx_mais_cara, col_m2])
        p_m2_base = float(tabela_zonas.loc[idx_mais_barata, col_m2])
        razao_valorizacao = round(p_m2_top / max(p_m2_base, 1.0), 2)

        vol_top_pct = float(tabela_zonas.loc[idx_maior_vol, "representatividade_pct"])
        vol_escasso_pct = float(tabela_zonas.loc[idx_menor_vol, "representatividade_pct"])
        cv_max = float(tabela_zonas.loc[idx_maior_disp, "coeficiente_variacao_pct"])

        diagnostico: dict[str, str] = {
            "zona_mais_valorizada": f"{zona_top} (R$ {p_m2_top:,.2f}/m²)",
            "zona_mais_acessivel": f"{zona_base} (R$ {p_m2_base:,.2f}/m²)",
            "fator_gradiente_espacial": f"{razao_valorizacao:.2f}x ({zona_top} vs {zona_base})",
            "zona_maior_liquidez": f"{zona_vol} ({vol_top_pct:.1f}% da base)",
            "zona_menor_amostragem": f"{zona_escassa} ({vol_escasso_pct:.1f}% da base)",
            "zona_maior_heterogeneidade": f"{zona_disp} (CV: {cv_max:.1f}%)",
            "implicacao_modelagem": (
                f"A localização geográfica exerce forte efeito multiplicador sobre o preço. "
                f"A diferença de {razao_valorizacao:.1f}x no metro quadrado entre {zona_top} e {zona_base} "
                f"justifica codificação One-Hot explícita e interações espaciais com a metragem."
            ),
        }

        if vol_escasso_pct < 5.0:
            diagnostico["alerta_amostragem"] = (
                f"A região '{zona_escassa}' possui menos de 5% da base ({vol_escasso_pct:.1f}%). "
                f"Recomenda-se cautela no serving para evitar extrapolamento preditivo nessa região."
            )

        return diagnostico

    def gerar_resumo_textual(self, tabela_zonas: pd.DataFrame, diagnostico: dict[str, str]) -> str:
        """Gera relatório descritivo completo em formato Markdown interpretando a dinâmica de Zonas.

        Parameters
        ----------
        tabela_zonas : pd.DataFrame
            DataFrame estruturado das zonas.
        diagnostico : dict[str, str]
            Diagnóstico sintetizado.

        Returns
        -------
        str
            Texto formatado em Markdown com análise mercadológica e estatística.
        """
        linhas: list[str] = [
            "# Interpretação da Análise Exploratória de Dados por Zona Imobiliária",
            "",
            "## 1. Destaques do Mercado por Região",
            f"- **Região Mais Valorizada**: {diagnostico.get('zona_mais_valorizada', 'N/D')}",
            f"- **Região Mais Acessível**: {diagnostico.get('zona_mais_acessivel', 'N/D')}",
            f"- **Gradiente de Valorização Espacial**: {diagnostico.get('fator_gradiente_espacial', 'N/D')}",
            f"- **Região com Maior Volume**: {diagnostico.get('zona_maior_liquidez', 'N/D')}",
            f"- **Região com Menor Volume**: {diagnostico.get('zona_menor_amostragem', 'N/D')}",
            f"- **Região com Maior Dispersão de Preços**: {diagnostico.get('zona_maior_heterogeneidade', 'N/D')}",
            "",
            "## 2. Implicações para o Modelo de Machine Learning",
            f"{diagnostico.get('implicacao_modelagem', '')}",
            "",
        ]

        if "alerta_amostragem" in diagnostico:
            linhas.extend([
                "### Alerta de Amostragem",
                f"> ⚠️ {diagnostico['alerta_amostragem']}",
                "",
            ])

        linhas.extend([
            "## 3. Tabela Comparativa de Indicadores por Zona",
            "",
            "| Zona | Imóveis | % Base | Preço Médio (R$) | Preço Mediano (R$) | R$/m² Médio | CV (%) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for _, row in tabela_zonas.iterrows():
            z = str(row.get("zona", ""))
            q = int(row.get("quantidade_imoveis", 0))
            pct = float(row.get("representatividade_pct", 0.0))
            p_med = float(row.get("preco_medio", 0.0))
            p_mdn = float(row.get("preco_mediano", 0.0))
            p_m2 = float(row.get("preco_m2_medio", 0.0))
            cv = float(row.get("coeficiente_variacao_pct", 0.0))
            linhas.append(
                f"| {z} | {q} | {pct:.1f}% | R$ {p_med:,.2f} | R$ {p_mdn:,.2f} | R$ {p_m2:,.2f} | {cv:.1f}% |"
            )

        linhas.append("")
        return "\n".join(linhas)

    def gerar_relatorio_negocio(
        self,
        tabela_zonas: pd.DataFrame,
        diagnostico: dict[str, str],
        stats_numericas: dict[str, dict[str, float | int]],
        outliers_iqr: dict[str, int],
        coluna_alvo: str = "Valor_da_Venda",
    ) -> str:
        """Gera relatório executivo em Markdown acessível à equipe de negócio.

        Traduz as métricas estatísticas da EDA para uma linguagem comercial clara,
        sem jargão técnico, com ênfase em impacto financeiro, riscos e oportunidades.

        Parameters
        ----------
        tabela_zonas : pd.DataFrame
            Tabela de estatísticas por Zona geográfica.
        diagnostico : dict[str, str]
            Diagnóstico de mercado gerado por gerar_diagnostico_zonas.
        stats_numericas : dict[str, dict[str, float | int]]
            Estatísticas descritivas das variáveis numéricas.
        outliers_iqr : dict[str, int]
            Quantidade de outliers detectados via IQR por coluna.
        coluna_alvo : str
            Nome da coluna de preço de venda.

        Returns
        -------
        str
            Relatório executivo em Markdown pronto para apresentação à liderança.
        """
        stats_valor = stats_numericas.get(coluna_alvo, {})
        preco_medio = float(stats_valor.get("mean", 0.0))
        preco_mediano = float(stats_valor.get("50%", stats_valor.get("median", 0.0)))
        preco_std = float(stats_valor.get("std", 0.0))
        preco_min = float(stats_valor.get("min", 0.0))
        preco_max = float(stats_valor.get("max", 0.0))
        total_imoveis = int(stats_valor.get("count", 0))

        cv_preco = round((preco_std / preco_medio) * 100.0, 1) if preco_medio > 0 else 0.0
        assimetria = preco_medio - preco_mediano

        zona_mais_valorizada = diagnostico.get("zona_mais_valorizada", "N/D")
        zona_mais_acessivel = diagnostico.get("zona_mais_acessivel", "N/D")
        gradiente = diagnostico.get("fator_gradiente_espacial", "N/D")
        zona_vol = diagnostico.get("zona_maior_liquidez", "N/D")
        zona_disp = diagnostico.get("zona_maior_heterogeneidade", "N/D")
        alerta_amostragem = diagnostico.get("alerta_amostragem", "")

        outliers_preco = outliers_iqr.get(coluna_alvo, 0)
        pct_outliers = round((outliers_preco / total_imoveis) * 100, 1) if total_imoveis > 0 else 0.0

        linhas: list[str] = [
            "# Relatório Executivo — Análise de Mercado Imobiliário",
            "",
            "> **Destinatário**: Equipe de Negócio e Liderança Comercial  ",
            "> **Finalidade**: Compreensão da base de dados utilizada no modelo preditivo de preços  ",
            "> **Fonte**: Análise Exploratória de Dados (EDA) — executada automaticamente antes do treinamento",
            "",
            "---",
            "",
            "## 🏠 Visão Geral do Mercado na Base de Dados",
            "",
            f"A base de dados analisada contém **{total_imoveis:,} imóveis** com as seguintes características de preço:",
            "",
            "| Indicador | Valor |",
            "| :--- | ---: |",
            f"| Preço médio de venda | R$ {preco_medio:,.2f} |",
            f"| Preço mediano de venda | R$ {preco_mediano:,.2f} |",
            f"| Menor preço registrado | R$ {preco_min:,.2f} |",
            f"| Maior preço registrado | R$ {preco_max:,.2f} |",
            f"| Variação típica em torno da média | R$ {preco_std:,.2f} |",
            "",
        ]

        # Interpretação da assimetria
        if abs(assimetria) > preco_mediano * 0.05:
            direcao = "acima" if assimetria > 0 else "abaixo"
            linhas += [
                "### 📊 Interpretação da Distribuição de Preços",
                "",
                f"O preço médio (R$ {preco_medio:,.2f}) está **{abs(assimetria):,.2f} {direcao} da mediana** "
                f"(R$ {preco_mediano:,.2f}). Isso indica que **imóveis de alto valor puxam a média para cima**, "
                f"sendo a mediana o indicador mais representativo do preço típico no mercado.",
                "",
            ]
        else:
            linhas += [
                "### 📊 Distribuição de Preços Equilibrada",
                "",
                f"A média e a mediana estão muito próximas (diferença de R$ {abs(assimetria):,.2f}), "
                f"indicando uma **distribuição de preços equilibrada** sem distorção significativa por imóveis atípicos.",
                "",
            ]

        # Heterogeneidade
        if cv_preco > 40:
            nivel_var = "muito alta"
            implicacao_var = (
                "Os preços variam de forma muito expressiva na base. "
                "Isso reflete um mercado com grande diversidade de perfis de imóveis e regiões. "
                "**O modelo precisa ser preciso nessa diferenciação** para evitar erros grandes de precificação."
            )
        elif cv_preco > 25:
            nivel_var = "moderada"
            implicacao_var = (
                "Existe variação relevante nos preços. "
                "A localização e as características físicas do imóvel têm peso significativo na precificação."
            )
        else:
            nivel_var = "baixa"
            implicacao_var = (
                "Os preços são relativamente homogêneos na base. "
                "O modelo tem um contexto mais previsível para operar."
            )

        linhas += [
            "### 📉 Heterogeneidade de Preços",
            "",
            f"A variação dos preços em relação à média é **{nivel_var}** ({cv_preco:.1f}% de coeficiente de variação).",
            f"{implicacao_var}",
            "",
        ]

        # Outliers
        linhas += [
            "### ⚠️ Imóveis com Preço Atípico",
            "",
            f"Foram identificados **{outliers_preco} imóveis com preço atípico** ({pct_outliers:.1f}% da base) "
            f"usando a regra estatística padrão do mercado (IQR).",
        ]
        if pct_outliers > 5:
            linhas += [
                "",
                "> 🔴 **Atenção**: A proporção de atípicos está acima de 5%. Recomenda-se revisão individual "
                "desses registros para verificar se são erros de cadastro ou imóveis realmente premium/especiais.",
            ]
        elif pct_outliers > 2:
            linhas += [
                "",
                "> 🟡 **Observação**: Proporção de atípicos dentro de um nível aceitável. "
                "O modelo tratará esses casos com regularização automática.",
            ]
        else:
            linhas += [
                "",
                "> 🟢 **Base saudável**: Poucos atípicos detectados. A base está bem comportada para precificação.",
            ]
        linhas.append("")

        # Análise por zona
        linhas += [
            "---",
            "",
            "## 🗺️ Análise por Região (Zona Geográfica)",
            "",
            f"- **Região de maior valor por m²**: {zona_mais_valorizada}",
            f"- **Região mais acessível**: {zona_mais_acessivel}",
            f"- **Fator de valorização entre extremos**: {gradiente} — ou seja, o metro quadrado da região mais cara "
            "custa esse múltiplo em relação à mais barata",
            f"- **Região com mais imóveis na base**: {zona_vol} — maior representatividade no treinamento do modelo",
            f"- **Região com maior variação interna de preços**: {zona_disp} — maior incerteza de precificação",
            "",
        ]

        if alerta_amostragem:
            linhas += [
                "> ⚠️ **Alerta Operacional**: " + alerta_amostragem,
                "",
            ]

        # Tabela de zonas
        if not tabela_zonas.empty and "zona" in tabela_zonas.columns:
            linhas += [
                "### Tabela de Referência de Preços por Região",
                "",
                "| Região | Imóveis | Preço Típico (Mediana) | Preço Mínimo | Preço Máximo | R$/m² Médio | Variação (%) |",
                "| :--- | :---: | ---: | ---: | ---: | ---: | :---: |",
            ]
            for _, row in tabela_zonas.iterrows():
                z = str(row.get("zona", ""))
                q = int(row.get("quantidade_imoveis", 0))
                p_mdn = float(row.get("preco_mediano", 0.0))
                p_min = float(row.get("preco_minimo", 0.0))
                p_max = float(row.get("preco_maximo", 0.0))
                p_m2 = float(row.get("preco_m2_medio", 0.0))
                cv = float(row.get("coeficiente_variacao_pct", 0.0))
                sinal_cv = "🔴" if cv > 40 else ("🟡" if cv > 25 else "🟢")
                linhas.append(
                    f"| {z} | {q:,} | R$ {p_mdn:,.2f} | R$ {p_min:,.2f} | R$ {p_max:,.2f} "
                    f"| R$ {p_m2:,.2f} | {sinal_cv} {cv:.1f}% |"
                )
            linhas.append("")

        # Perfil físico por zona
        if not tabela_zonas.empty and "metragem_media" in tabela_zonas.columns:
            linhas += [
                "### Perfil Físico Médio dos Imóveis por Região",
                "",
                "| Região | Metragem Média (m²) | Quartos | Banheiros | Vagas |",
                "| :--- | :---: | :---: | :---: | :---: |",
            ]
            for _, row in tabela_zonas.iterrows():
                z = str(row.get("zona", ""))
                mt = float(row.get("metragem_media", 0.0))
                qt = float(row.get("quartos_medio", 0.0)) if "quartos_medio" in tabela_zonas.columns else 0.0
                bn = float(row.get("banheiros_medio", 0.0)) if "banheiros_medio" in tabela_zonas.columns else 0.0
                vg = float(row.get("vagas_medio", 0.0)) if "vagas_medio" in tabela_zonas.columns else 0.0
                linhas.append(f"| {z} | {mt:.1f} m² | {qt:.1f} | {bn:.1f} | {vg:.1f} |")
            linhas.append("")

        # Recomendações executivas
        linhas += [
            "---",
            "",
            "## 💡 Recomendações para a Equipe Comercial",
            "",
            f"1. **Use a mediana regional como referência de preço**, não a média — "
            f"ela é menos afetada por imóveis atípicos e reflete melhor o valor de mercado típico.",
            f"2. **Ao avaliar um imóvel**, compare com imóveis da mesma zona, pois o gradiente de "
            f"{gradiente} mostra que localização é o fator de maior impacto no preço por m².",
            f"3. **Em negociações na região '{zona_disp.split(' ')[0]}'**, use margens de negociação mais "
            f"amplas — é a zona com maior variação interna de preços, logo a incerteza é maior.",
            f"4. **Para decisões de portfólio**, priorize imóveis na região '{zona_vol.split(' ')[0]}' "
            f"onde há maior volume de dados — o modelo preditivo terá maior precisão nessa região.",
            "",
            "---",
            "",
            "*Relatório gerado automaticamente pela plataforma de Machine Learning — Previsão de Preço de Imóveis.*",
        ]

        return "\n".join(linhas)

