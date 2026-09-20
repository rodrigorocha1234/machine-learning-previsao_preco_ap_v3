# Spec — Carregamento e Validação de Dados

## Responsabilidades
Criar a classe `DataLoader` e o componente `DatasetValidator`.

## Contrato de entrada
O dataframe deve possuir exatamente, no mínimo, as colunas:

| Coluna | Tipo lógico | Regra |
|---|---|---|
| Zona | categórica/string | não vazia |
| Quartos | inteiro | >= 0 |
| Banheiros | inteiro | >= 0 |
| Vagas_Garagem | inteiro | >= 0 |
| Metragem | numérico | > 0 |
| Valor_da_Venda | numérico | > 0 |

## DataLoader
### Interface
```python
class DataLoader(Protocol):
    def load(self) -> pd.DataFrame: ...
```

### Implementações previstas
- `CsvDataLoader`
- `ParquetDataLoader`

### Regras
- preservar nomes originais das colunas;
- converter espaços externos em strings;
- não fazer imputação no carregador;
- registrar quantidade de linhas e colunas por evento de domínio;
- não eliminar outliers silenciosamente.

## DatasetValidator
### Validações obrigatórias
- colunas obrigatórias;
- tipos coerentes;
- target positivo;
- metragem positiva;
- duplicatas completas;
- percentual de nulos por coluna;
- categorias vazias em `Zona`;
- valores inteiros negativos em quartos, banheiros e vagas;
- linhas com preço/m² impossível segundo limites configuráveis.

## Política para duplicatas
Duplicatas exatas devem ser reportadas. A remoção é configurável e deve aparecer no relatório de EDA.

## Critérios de aceite
- dataset inválido gera exceção de domínio com lista de problemas;
- dataset válido retorna dataframe sem alterar o target;
- todas as decisões de limpeza ficam registradas como metadados reproduzíveis.
