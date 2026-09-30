# Rules 02 — Controle de Fluxo sem `if`

## Regra
Código de produção próprio não deve utilizar `if`, `elif` ou `match` para ramificações de domínio, arquitetura ou seleção de implementação.

## Substituições preferenciais
- Strategy: algoritmos intercambiáveis;
- State: comportamento dependente de estado;
- Specification: regras booleanas combináveis;
- Chain of Responsibility: validações sequenciais;
- Null Object: eliminar verificações de ausência;
- Factory/Abstract Factory: criação configurável;
- Command: ações selecionadas dinamicamente;
- Registry/dispatch dict: seleção por chave YAML;
- polimorfismo Protocol/ABC: comportamento por tipo;
- configuração declarativa e composição de objetos.

## Regra de simplicidade
Não trocar um `if` trivial por uma arquitetura exagerada. Quando uma condição inevitável vier de API externa ou biblioteca, encapsular no menor adapter possível e documentar em `Decision_Log.md`. O domínio não deve propagar condicionais.
