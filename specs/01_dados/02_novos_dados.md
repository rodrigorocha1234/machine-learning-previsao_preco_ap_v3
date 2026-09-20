# Alimentação com Novos Dados

Existem dois fluxos distintos.

## 1. Inferência

Novos imóveis sem `Valor_da_Venda` são enviados ao modelo servido pelo MLflow.

A pipeline reutiliza exatamente o pré-processamento persistido com o modelo.

## 2. Aprendizado / retreinamento

Quando o valor real da venda ficar disponível:

1. armazenar o novo registro;
2. validar schema;
3. executar qualidade;
4. avaliar drift;
5. anexar ao dataset de treino versionado;
6. abrir nova execução de Grid Search + validação cruzada;
7. comparar novo candidato com o campeão;
8. promover somente se passar os gates técnicos e de negócio.

Nunca executar `partial_fit` implicitamente em modelos que não suportam aprendizagem incremental.

## Gate mínimo para promoção

- schema válido;
- drift conhecido;
- nenhuma regressão material nas métricas obrigatórias;
- Friedman/Nemenyi reexecutados quando múltiplos candidatos são comparados;
- holdout final preservado;
- artefatos completos no MLflow.
