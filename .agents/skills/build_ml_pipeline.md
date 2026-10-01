# Skill — Build ML Pipeline

## Escopo
Implementar preprocessamento, modelos, Nested CV, tuning, avaliação, estatística, seleção e ensemble.

## Regra estatística central
- INNER CV escolhe hiperparâmetros.
- OUTER CV estima generalização e alimenta comparação estatística.
- HOLDOUT é aberto uma única vez após congelar a configuração final.

## Modelos obrigatórios
Linear Regression, Lasso, Ridge, Elastic Net, Decision Tree Regressor, Random Forest Regressor, SVR, MLPRegressor, XGBoost Regressor e LightGBM Regressor.

## Padrões
- Strategy para modelo;
- Strategy para preprocessamento;
- Strategy para tuning;
- Factory/registry para criação;
- Null Object para `TuningNenhum`;
- Policy/Specification para seleção;
- Strategy para ensemble.

## YAML
Cada modelo possui:
- `parametros` para construção base;
- `tuning.estrategia`;
- `tuning.parametros` ou distribuições;
- `tuning.n_iter`, scoring e random_state quando aplicável.

## MLflow após tuning
Para cada tuning interno e tuning final registrar:
- estratégia;
- espaço pesquisado como artifact;
- best params;
- best score;
- duração;
- quantidade de tentativas;
- falhas;
- resultados de CV;
- versão do YAML;
- modelo e preprocessamento.

## Interpretação para negócio
Após o treinamento de cada modelo, produzir artifact Markdown/JSON contendo cada parâmetro efetivamente usado e uma explicação de negócio simples, por exemplo:
- `alpha`: intensidade de regularização e efeito sobre estabilidade/complexidade;
- `max_depth`: profundidade permitida e trade-off entre regras simples e ajuste fino;
- `n_estimators`: quantidade de árvores e efeito esperado em estabilidade/custo;
- `C`: tolerância do SVR à penalização;
- `hidden_layer_sizes`: capacidade da rede;
- `learning_rate`: velocidade de atualização do boosting/rede.

A interpretação não pode afirmar causalidade onde existe apenas configuração técnica.


## Referência para o projeto existente

Este arquivo descreve o processo de autoria do Antigravity; seus objetivos não são evidências de implementação ou homologação. Para operar a aplicação existente, consulte o [índice atual da documentação](../../docs/README.md). Para mudanças, confira primeiro o [estado de atendimento dos requisitos](../../production_artifacts/Technical_Specification.md). O workflow `/startcycle` continua reservado à sua invocação explícita, com os gates definidos naquele fluxo.

A implementação atual já agrega média, mediana e desvio padrão (`ddof=0`) das seis métricas externas. O comitê selecionado ainda não é um ensemble preditivo; a documentação deve manter essa pendência explícita.
