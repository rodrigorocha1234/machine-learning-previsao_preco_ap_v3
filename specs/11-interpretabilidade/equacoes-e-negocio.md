# Spec — Equações, coeficientes e interpretação

## Saída obrigatória para lineares
CSV:
- feature transformada;
- coeficiente;
- sinal;
- magnitude;
- unidade interpretável quando possível;
- categoria de referência;
- texto de negócio.

Exemplo:
`Metragem: +R$ 4.200 por m²`, se o coeficiente estiver recuperado para escala original e o modelo for realmente linear nessa variável.

Se o pipeline estiver padronizado:
- registrar coeficiente padronizado;
- quando possível, converter para escala original;
- não interpretar diretamente um coeficiente padronizado como “reais por unidade”.

## Categóricas
Com `drop="first"`:
`Zona_Sul = +R$ 80.000` significa diferença estimada em relação à categoria de referência, mantendo demais variáveis constantes.

## Não-lineares
Registrar:
- estrutura matemática;
- permutation importance;
- SHAP opcional;
- partial dependence;
- exemplos contrafactuais simples.

## Polinomial
Nunca resumir um termo `x²` como efeito constante. Calcular derivada/marginal:
`dŷ/dx = β1 + 2β2x + ...`
