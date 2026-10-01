# Revisão final — estado e pendências

**Estado: implementação funcional com validações parciais; homologação integral não demonstrada.**

Este documento substitui o parecer anterior de “100% aprovado para produção”. A revisão do código encontrou divergências entre aquele texto e a implementação. Não são mantidas como evidência métricas fixas de campeão, números de runs, contagens de arquivos ou assinaturas de agentes sem comprovação atual.

## O que está presente

- Pipeline de regressão, pré-processamento dentro das buscas e CV aninhada.
- VotingRegressor final com tuning por integrante, média de pesos iguais e serialização no pyfunc.
- Desvio padrão descritivo para seis métricas, com registro MLflow/Prometheus e exibição por modelo no Grafana.
- Avaliação por localidade e resposta pyfunc com referências de negócio.
- Serving nativo MLflow instrumentado, com versão fixa durante o processo.
- Snapshot persistente e exportador independente para manter resultados de treino após encerramento do pipeline.
- Testes direcionados e verificações operacionais, com escopo no [QA Report](QA_Report.md).

## O que não deve ser afirmado

Não há evidência de holdout criptografado, estratificação efetiva do split, target transformado com log1p, conformidade integral com todas as regras, drift real contínuo integrado ou erro de produção associado a vendas reais.

O modelo campeão, sua versão, seus scores e a quantidade de runs são fatos de cada ambiente/execução. Consulte o Registry, o run associado e a versão efetivamente carregada no serving. Não use valores do monitor demonstrativo como resultados de homologação.

## Pendências para uma avaliação de prontidão

| Área | Pendência |
| --- | --- |
| Estatística | Avaliar pressupostos da comparação, dependência entre folds e grupos com poucas observações |
| Isolamento | Fortalecer controles hoje baseados em asserts/chave constante e documentar acessos aos dados brutos |
| Seleção | Avaliar desempenho do VotingRegressor na base real; a CV atual mede os componentes, não o conjunto selecionado |
| Negócio | Integrar suficiência ao caminho vetorizado e externalizar percentuais pretendidos |
| Rastreabilidade | Persistir integralmente tuning, configurações e parâmetros efetivos |
| Dados | Implementar fontes/EDA histórica previstas e baseline real de drift |
| Engenharia | Auditoria global de lint, tipos, padrões de código e suíte completa |
| Operação | Fixar dependências, reduzir instalações no startup, verificar backup, autenticação, TLS e rede |
| Métricas | Definir estratégia para múltiplos workers e múltiplos produtores de snapshot antes de escalar |

O Compose atual usa configurações permissivas e não é apresentado como implantação pública endurecida. A documentação atualizada não modifica nem aprova automaticamente essas configurações.

## Referências

[Arquitetura](Architecture.md) · [Especificação](Technical_Specification.md) · [Decisões](Decision_Log.md) · [Operação](Deployment.md) · [QA](QA_Report.md).
