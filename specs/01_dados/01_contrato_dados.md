# Contrato de Dados

## Schema mínimo

```yaml
Zona:
  tipo: categoria
  obrigatorio: true
Quartos:
  tipo: inteiro
  minimo: 0
Banheiros:
  tipo: inteiro
  minimo: 0
Vagas:
  tipo: inteiro
  minimo: 0
Metragem:
  tipo: float
  minimo_exclusivo: 0
Valor_da_Venda:
  tipo: float
  minimo_exclusivo: 0
```

## Novas variáveis

Novos atributos devem entrar primeiro no contrato, depois na EDA e somente então no pipeline.

## Valores ausentes

- categóricas: imputação por moda ou categoria explícita `DESCONHECIDO`;
- numéricas: mediana por padrão;
- regras especiais devem ser configuráveis.

## Categorias desconhecidas em produção

`OneHotEncoder(handle_unknown="ignore")`.

## Duplicatas

Definir chave de negócio quando disponível. Sem chave, duplicatas perfeitas serão reportadas e não eliminadas silenciosamente.
