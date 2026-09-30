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
