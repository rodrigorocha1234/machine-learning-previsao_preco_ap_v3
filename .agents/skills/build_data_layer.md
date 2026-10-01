# Skill — Build Data Layer

## Escopo
Criar camada de carregamento, staging, contratos, EDA, EDA histórica e drift.

## Carregadores
Definir `ProtocoloCarregador[T_co]` com TypeVar covariante e implementações separadas por fonte. Seleção via Factory/registry a partir do YAML.

## Staging
Repository + Adapter, inicialmente SQLite, substituível sem alterar orquestração.

## Pandas
Proibidos loops linha a linha. Priorizar vetorização, `assign`, `where`, `mask`, `groupby`, `agg`, `transform`, `merge`, `join`, `numpy.select`.

## Funcional
Usar `map`, `filter`, `reduce`, `zip`, `zip_longest`, `itertools.chain`, `itertools.product`, `itertools.pairwise` e afins somente quando melhorarem a expressão do algoritmo. Nunca trocar uma operação pandas vetorizada por Python funcional.


## Referência para o projeto existente

Este arquivo descreve o processo de autoria do Antigravity; seus objetivos não são evidências de implementação ou homologação. Para operar a aplicação existente, consulte o [índice atual da documentação](../../docs/README.md). Para mudanças, confira primeiro o [estado de atendimento dos requisitos](../../production_artifacts/Technical_Specification.md). O workflow `/startcycle` continua reservado à sua invocação explícita, com os gates definidos naquele fluxo.
