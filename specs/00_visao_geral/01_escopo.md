# Escopo do Projeto

## Problema

Estimar o valor de venda de um imóvel a partir de atributos cadastrais, físicos e de localização.

## Saídas

1. `Valor_Previsto`
2. `Valor_Com_Desconto`
3. `Percentual_Desconto`
4. intervalo/faixa de segurança quando habilitado
5. identificação da versão do modelo
6. métricas técnicas
7. métricas da imobiliária
8. evidências de drift
9. explicações do modelo

## Fora do escopo inicial

- precificação jurídica;
- avaliação oficial para financiamento;
- substituição de avaliação humana;
- recomendação automática de desconto sem limites de negócio;
- atualização automática do campeão sem gates de validação.

## Princípios

- reprodutibilidade;
- separação entre regra técnica e regra de negócio;
- nenhuma regra de desconto embutida no algoritmo de regressão;
- rastreabilidade no MLflow;
- compatibilidade com processamento distribuído futuro.
