# Limites de Drift

Os limites são configuráveis e devem ser calibrados com histórico.

Exemplo inicial para PSI:

```text
PSI < 0.10       → estável
0.10 ≤ PSI < .25 → atenção
PSI ≥ 0.25       → drift forte
```

Essas faixas são heurísticas e não substituem validação no contexto da imobiliária.

Para testes com p-valor, registrar também tamanho de efeito. Com datasets grandes, diferenças pequenas podem se tornar estatisticamente significativas sem relevância prática.

A decisão de retreinar deve combinar:

- drift;
- degradação real de erro;
- volume de novos dados;
- mudança de regras de negócio;
- mudança estrutural do mercado.
