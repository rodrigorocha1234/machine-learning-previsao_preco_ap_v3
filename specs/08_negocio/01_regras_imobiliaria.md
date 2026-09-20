# Regras de Negócio da Imobiliária

Este diretório é independente das specs de Machine Learning.

## Princípio

O modelo prevê valor. A regra de negócio decide como usar essa previsão.

## Regra inicial — desconto

Entrada:

- `Valor_Previsto`
- `Percentual_Desconto`

Cálculo:

\[
Valor\_Com\_Desconto = Valor\_Previsto \times (1 - Percentual\_Desconto/100)
\]

## Limites

Configurar:

- desconto mínimo;
- desconto máximo permitido;
- desconto que exige aprovação humana;
- desconto proibido.

Exemplo de configuração, não valor normativo:

```yaml
desconto:
  minimo: 0
  maximo_automatico: 5
  maximo_com_aprovacao: 10
```

Esses valores devem ser definidos pela imobiliária e podem mudar sem retreinar o modelo.
