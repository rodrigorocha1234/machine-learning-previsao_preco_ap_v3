# Skill — Design Architecture

## Objetivo
Traduzir a especificação aprovada em arquitetura Python modular e testável.

## Regras
- Python 3.12+;
- strict typing;
- proibido `Any`;
- Protocol para dependências substituíveis;
- TypeVar e Generic para componentes reutilizáveis;
- propriedades com `@property` e `@setter` quando houver invariantes ou encapsulamento real;
- uma classe por arquivo, exceto dataclasses, enums coesos e exceções customizadas;
- nomes próprios em português;
- módulos/pacotes com exatamente duas palavras separadas por `_`;
- não sombrear bibliotecas externas;
- composição antes de herança;
- `@override` quando aplicável.

## Decisões sem if
Não usar `if`, `elif` ou `match` para decisões de domínio/arquitetura. Escolher entre:
- Strategy;
- State;
- Specification;
- Chain of Responsibility;
- Null Object;
- Factory;
- Command;
- Registry/dispatch table;
- polimorfismo por Protocol/ABC;
- configuração declarativa.

## Entrega
- `production_artifacts/Architecture.md`
- árvore alvo em `app_build/`.


## Referência para o projeto existente

Este arquivo descreve o processo de autoria do Antigravity; seus objetivos não são evidências de implementação ou homologação. Para operar a aplicação existente, consulte o [índice atual da documentação](../../docs/README.md). Para mudanças, confira primeiro o [estado de atendimento dos requisitos](../../production_artifacts/Technical_Specification.md). O workflow `/startcycle` continua reservado à sua invocação explícita, com os gates definidos naquele fluxo.
