# Spec — Schema e qualidade

## Schema mínimo
| Coluna | Tipo lógico | Regra |
|---|---|---|
| Zona | categoria/string | obrigatória, não vazia |
| Quartos | inteiro >= 0 | obrigatória |
| Banheiros | inteiro >= 0 | obrigatória |
| Vagas | inteiro >= 0 | obrigatória |
| Metragem | float > 0 | obrigatória |
| Valor_da_Venda | float > 0 | obrigatória em treino |

## Regras
1. Categorias desconhecidas na inferência devem ser toleradas com `OneHotEncoder(handle_unknown="ignore")`.
2. Nulos numéricos: `SimpleImputer(strategy="median")`.
3. Nulos categóricos: `SimpleImputer(strategy="most_frequent")`.
4. Outliers não devem ser removidos automaticamente sem rastreamento. O relatório deve marcar IQR, z-score robusto/MAD e extremos.
5. O split de holdout ocorre antes de qualquer ajuste de transformadores.
6. Transformadores são ajustados somente nos dados de treino de cada fold.

## Feature engineering opcional
- `Preco_por_m2` somente para análise histórica/EDA; **não usar como entrada** se for calculado com `Valor_da_Venda`, pois causaria leakage.
- `Densidade_quartos = Quartos / Metragem`.
- `Densidade_banheiros = Banheiros / Metragem`.
- `Total_comodos_chave = Quartos + Banheiros`.
- `Tem_vaga = Vagas > 0`.
