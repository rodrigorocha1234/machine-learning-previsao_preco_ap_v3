# Coeficientes e Interpretação

## Modelos lineares

Extrair:

- nome da feature após transformação;
- coeficiente;
- sinal;
- magnitude;
- coeficiente padronizado quando houver;
- interpretação automática preliminar.

Exemplo:

```text
Feature: Metragem
Coeficiente: 1.580,25
Interpretação:
mantidas as demais variáveis constantes, uma unidade adicional de metragem
está associada a aproximadamente R$ 1.580,25 adicionais no preço previsto,
na escala original do atributo.
```

## One-hot de Zona

Exemplo:

```text
Zona_Centro = +85.000
```

Interpretação: em relação à categoria de referência implícita/estrutura de codificação, imóveis nessa categoria recebem ajuste médio associado ao coeficiente, mantendo as demais variáveis constantes.

## Modelos não lineares

Não fabricar “coeficientes” inexistentes. Registrar:

- feature importance;
- permutation importance;
- SHAP opcional;
- regras/árvores;
- efeitos parciais.
