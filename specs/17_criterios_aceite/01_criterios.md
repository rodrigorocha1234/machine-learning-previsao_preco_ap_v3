# Critérios de Aceite

O projeto só está completo quando:

- [ ] EDA é gerada antes do treino;
- [ ] categorias são detectadas e tratadas;
- [ ] scaler é escolhido conforme a família do modelo;
- [ ] cada classe está em módulo próprio;
- [ ] Grid Search ocorre antes da validação comparativa;
- [ ] Grid Search é registrado no MLflow;
- [ ] `RepeatedKFold` usa 30 repetições;
- [ ] mesmas divisões são usadas para todos os modelos;
- [ ] Friedman recebe apenas resultados pós-CV;
- [ ] Nemenyi só roda quando Friedman é significativo;
- [ ] grupo estatisticamente elegível é salvo;
- [ ] votação usa somente modelos elegíveis quando `usar_votacao=true`;
- [ ] com `usar_votacao=false`, nenhum ensemble é criado;
- [ ] o estado `ensemble/usado` é registrado no MLflow;
- [ ] campeão é treinado e avaliado no holdout;
- [ ] métricas imobiliárias são calculadas para o campeão;
- [ ] coeficientes/equações são registrados quando aplicáveis;
- [ ] explicabilidade substitui “coeficientes” em modelos não lineares;
- [ ] curva de aprendizado é registrada diretamente no MLflow;
- [ ] DataFrames, figuras, dicionários e textos são enviados diretamente ao MLflow sem persistência local permanente;
- [ ] data drift é implementado;
- [ ] PyFunc aceita `Percentual_Desconto`;
- [ ] serving usa MLflow e não API própria da aplicação;
- [ ] entrada de novos dados possui fluxo de inferência e retreinamento;
- [ ] interface permite backend distribuído futuro.


- [ ] `VotingRegressor` do sklearn é usado nas estratégias de voting;
- [ ] `StackingRegressor` do sklearn é usado no stacking;
- [ ] `BaggingRegressor` do sklearn é usado no bagging;
- [ ] `GradientBoostingRegressor` está disponível como candidato;
- [ ] `CatBoostRegressor` está disponível como candidato;
- [ ] uma classe por arquivo `.py` também é respeitada nas implementações de ensemble;

- [ ] todos os packages dentro de `src/imobiliaria_ml/` usam o sufixo `_pkg`;

- [ ] nenhum arquivo Python de produção utiliza `typing.Any` ou `Any`;
- [ ] payloads de eventos do Observer são tipados com `TypedDict`, `dataclass` ou tipos concretos;
- [ ] configurações não usam `dict[str, Any]`;

- [ ] `TipoModelo` é usado nas factories de modelos;
- [ ] `TipoEvento` é usado no Observer em vez de strings livres;
- [ ] `TipoEscalonador` é usado na seleção de scalers;
- [ ] `TipoEnsemble` e `OrigemPesoVoting` são usados no módulo de ensemble;
- [ ] `NivelDrift` é usado na classificação de drift;
- [ ] `TipoCarregador` é usado na factory de carregadores;
- [ ] `TipoMetrica` é usado na seleção de métricas internas;
- [ ] não existem strings mágicas para categorias cobertas pelos enums definidos;

- [ ] `Generic` e `TypeVar` são usados quando necessário para preservar tipos;
- [ ] `TypeVar` usa `bound` ou constraints quando houver contrato conhecido;
- [ ] `Protocol` é preferido como bound para contratos comportamentais;
- [ ] `TypeVar` não é usado apenas para mascarar ausência de tipagem;
- [ ] Strategies de modelos preservam o tipo concreto do estimador;
- [ ] Executor/Adapter distribuído usa generics tipados;
