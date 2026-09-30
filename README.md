# Projeto Multiagente — Previsão de Preços de Apartamentos

Workspace de Context-as-Code para Google Antigravity, estruturado para gerar o projeto de previsão de preços de apartamentos de Ribeirão Preto/SP.

## Estrutura
- `.agents/agents.md`: personas e responsabilidades.
- `.agents/skills/`: capacidades especializadas.
- `.agents/workflows/startcycle.md`: orquestrador `/startcycle`.
- `specs/`: especificações funcionais e técnicas.
- `rules/`: regras obrigatórias de implementação.
- `configs/`: configuração inicial de pipeline e tuning.
- `production_artifacts/`: handoff entre agentes.
- `app_build/`: código final gerado pelo Antigravity.

## Disparo
No Gerenciador de Agentes do Antigravity:

```text
/startcycle Implementar o projeto de previsão de preços de apartamentos de Ribeirão Preto/SP conforme specs, rules e configs deste workspace.
```

O fluxo deve parar depois da especificação e aguardar `Approved` antes da geração de código.
