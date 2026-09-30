# Registro de Decisões Arquiteturais (Decision Log)

**Autor:** @ml_architect (Arquiteto de Machine Learning)  
**Data:** 2026-09-29  
**Status:** Aprovado  

---

## Decisão 01: Eliminação de `if`/`elif`/`match` no Domínio via Registry e Dispatch Tables
- **Contexto:** A especificação exige ausência de estruturas condicionais proceduralmente ramificadas no código próprio.
- **Decisão:** Toda seleção de algoritmos, modelos, transformadores e estratégias de busca é mapeada em tabelas declarativas de despacho (`Mapping[str, Callable]`). A resolução ocorre por indexação direta de dicionário tipado `REGISTRO[chave]`.
- **Consequência:** Eliminação total de `if/elif` na seleção de componentes, facilidade para plugar novos estimadores ou estratégias sem alterar classes orquestradoras (Princípio Aberto-Fechado).

---

## Decisão 02: Padrão Null Object para Modelos sem Tuning
- **Contexto:** Modelos como a Regressão Linear Simples possuem `tuning.estrategia: nenhum`.
- **Decisão:** Em vez de condicionais do tipo `if modelo.tuning:`, implementamos `EstrategiaNula` herdando da mesma interface `ContratoTuning`, que simplesmente retorna o estimador ajustado diretamente no conjunto de treino.
- **Consequência:** Polimorfismo limpo e uniforme ao longo de todo o loop de Nested CV.

---

## Decisão 03: Desacoplamento de MLflow via Observer Pattern
- **Contexto:** Acoplar chamadas diretas da API `mlflow` dentro de estimadores ou orquestradores de validação cruzada polui a lógica de domínio e dificulta testes unitários.
- **Decisão:** Os componentes de validação emitem eventos imutáveis (`EventoFoldConcluido`, `EventoTuningConcluido`). O `ObservadorMlflow` assina o despachante de eventos e persiste parâmetros, métricas e artefatos no servidor remoto.
- **Consequência:** A esteira de machine learning opera de forma autônoma e pode rodar em modo headless/offline com `ObservadorNulo` em testes automatizados.

---

## Decisão 04: Isolamento Criptográfico e Lógico de Holdout
- **Contexto:** Risco de data leakage pelo acesso antecipado ao conjunto de holdout durante a exploração ou tuning.
- **Decisão:** Implementação da classe `CofreHoldout` que encapsula o particionamento e requer uma chave de liberação vinculada ao estado imutável `EstadoPipeline.CONGELADO`. Tentativas de acesso antes da etapa 15 disparam a exceção `ViolacaoIsolamentoDadosErro`.
- **Consequência:** Garantia arquitetural de que o conjunto de teste definitivo jamais vazará para folds internos ou externos.

---

## Decisão 05: Hierarquia Estrita GLOBAL → ZONA → BAIRRO com Fallback Polimórfico
- **Contexto:** Requisito de negócio exigindo que as métricas imobiliárias e a suficiência amostral sejam sempre computadas e avaliadas nessa ordem exata.
- **Decisão:** O cálculo é feito por um agregador hierárquico com padrão *Specification*. Se a amostragem do Bairro for inferior a 20, o objeto de resultado automaticamente delega o cálculo das faixas de segurança para as estatísticas da Zona; se a Zona for inferior a 30, delega para o Global.
- **Consequência:** Atendimento estrito à regra de negócio imobiliária sem uso de condicionais aninhadas no código de aplicação.
