# Rules 03 — Pandas e estilo funcional

## Proibido para processamento tabular
- iterrows()
- itertuples()
- apply(axis=1)
- loop sobre registros
- loop sobre índices
- atribuição célula a célula

## Prioridade
vetorização → Series.str/dt/cat → assign/where/mask → numpy.where/select → groupby.agg/transform → merge/join → isin/between → fillna/replace/clip.

## Funções Python
`map`, `filter`, `functools.reduce`, `zip`, `itertools.zip_longest`, `chain`, `product`, `pairwise` e demais utilitários podem ser usados para coleções Python, configurações, modelos, folds, combinações e orquestração. Não devem substituir operações vetorizadas do pandas.
