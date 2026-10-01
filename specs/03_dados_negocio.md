# Spec 03 — Dados e regras de negócio

## Requisitos

Alvo `Valor_da_Venda`, avaliação GLOBAL/ZONA/BAIRRO. Separar previsão técnica de indicadores de referência, descontos e faixas. Limites de amostra por YAML; estados `SUFICIENTE`, `AMOSTRA_INSUFICIENTE` e `NAO_DISPONIVEL`.

Fontes previstas: CSV, Parquet, JSON, SQLite, PostgreSQL, REST, S3 e Spark. Configuração declarativa de descontos e fallback por suficiência são requisitos desejados.

## Estado atual

Carregadores Excel/CSV/Parquet disponíveis; a etapa principal usa explicitamente `dados/bairro_final_v3_engineered.xlsx`. Staging em SQLite substitui a tabela do snapshot. As demais fontes não estão integradas ao fluxo.

As estatísticas geográficas são construídas com a base de desenvolvimento. Limites atuais: 30 imóveis por zona e 20 por bairro. O caminho de previsão individual dispõe de resolução de suficiência; o enriquecimento vetorizado usado pela API faz fallback por ausência de localidade.

A API devolve preço individual, R$/m² e referências de mercado. `valor_previsto_zona` e `valor_previsto_bairro` são médias do lote recebido, não previsões contrafactuais do mesmo imóvel em outros locais. Bairros no agrupamento do lote usam a combinação Zona/Bairro; a tabela histórica de referências é indexada apenas por Bairro.

Descontos de 5%, 10% e 15% e faixas atuais são constantes no código. No caminho vetorizado, usam medianas históricas por nível. Não são intervalos estatísticos nem garantias de negociação. [Contrato da API](../docs/exemplo_chamada_api_mlflow.md).
