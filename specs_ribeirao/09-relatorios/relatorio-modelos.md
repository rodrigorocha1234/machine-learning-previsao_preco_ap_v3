# Spec — Relatório de Modelos e Seleção

## Nome sugerido
`model_report.md`

## Estrutura
### 1. Resumo executivo
- campeão;
- MAE/RMSE/R² médios;
- nível de incerteza;
- justificativa estatística.

### 2. Protocolo experimental
Explicar nested CV: 30 folds externos + Grid Search interno.

### 3. Tabela comparativa
Por modelo:
- MAE médio/mediana/desvio;
- RMSE;
- R²;
- MAPE;
- rank médio;
- tempo de treino.

### 4. Friedman
Estatística, p-value, interpretação.

### 5. Nemenyi
Matriz de p-values e pares significativamente diferentes.

### 6. Hiperparâmetros
Tabela com os melhores parâmetros do refit final de cada modelo.

### 7. Equações/funções
Incluir conteúdo exigido em `equacoes-dos-modelos.md`.

### 8. Interpretação de negócio
Explicar coeficientes/importâncias, sem linguagem causal.

### 9. Seleção do campeão
Demonstrar a regra aplicada, sem decisão manual não rastreada.

### 10. Limitações
Cobertura geográfica, amostra, drift, ausência de atributos importantes (andar, condomínio, idade, localização exata etc.).
