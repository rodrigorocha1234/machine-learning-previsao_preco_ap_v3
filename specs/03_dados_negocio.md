# Spec 03 — Dados e Regras de Negócio

## Domínio
Target: `Valor_da_Venda`.
Grupos de avaliação: `Zona` e `Bairro`.

## Fontes substituíveis
CSV, Parquet, JSON, SQLite, PostgreSQL, REST, S3/object storage e Spark.

## Regras imobiliárias
Calcular separadamente da previsão técnica:
- valor previsto;
- valor/m² previsto;
- valor/m² médio e mediano por Zona;
- índice da Zona;
- diferença percentual para média;
- descontos configuráveis;
- faixa segura de compra conforme regras configuradas.

## Amostra
Estados de domínio: `SUFICIENTE`, `AMOSTRA_INSUFICIENTE`, `NAO_DISPONIVEL`. Limites por Zona/Bairro vêm do YAML.
