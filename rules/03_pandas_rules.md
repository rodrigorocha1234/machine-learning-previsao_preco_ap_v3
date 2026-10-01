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


## Escopo e verificação

Este documento define requisitos de desenvolvimento; não comprova que todo o código atual já esteja conforme. O estado conhecido e as divergências estão na [especificação técnica](../production_artifacts/Technical_Specification.md) e no [relatório de qualidade](../production_artifacts/QA_Report.md). A atualização da documentação não flexibiliza estas regras nem substitui sua auditoria.
