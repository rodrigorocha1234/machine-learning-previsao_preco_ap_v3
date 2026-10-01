# Spec 02 — Modelos e tuning

## Requisitos

Suportar Linear Regression, Ridge, Lasso, Elastic Net, Decision Tree, Random Forest, SVR, MLPRegressor, XGBoost e LightGBM. Escolher estratégia por YAML: `grid`, `random` ou `nenhum`; permitir extensão sem acoplar novas buscas ao fluxo principal.

Persistir parâmetros efetivos, scores, configuração, histórico de tuning interno/final e explicação de parâmetros para negócio. Esses requisitos não equivalem a afirmar que toda a persistência já existe.

## Estado atual

As dez famílias estão na fábrica. Ridge, árvore de decisão e Random Forest estão ativos na configuração atual. As buscas usam pipeline completo, `refit=True` e `n_jobs=None`; o paralelismo do estimador é independente.

O run pai `nested_cv_<modelo>` registra médias, desvios padrão, algumas medianas, `desvio_padrao_ddof=0` e quantidade de folds. Runs filhos registram métricas externas e melhores parâmetros. O histórico completo da busca, todos os parâmetros efetivos e o tuning final ainda não têm rastreabilidade integral.

A opção de votação identifica um grupo e seu líder; o fluxo final treina apenas o líder, sem VotingRegressor. [Guia de modelagem](../docs/validacao_cruzada_e_tuning.md).
