# Relatório de qualidade — escopo verificado

Este relatório substitui afirmações anteriores de aprovação integral que não são sustentadas pelo estado atual do código. Não certifica ausência global de condicionais, tipagem estrita completa ou conformidade de todos os requisitos.

## Testes executados nesta revisão

```bash
.venv/bin/python -m pytest \
  tests/test_dispersao_cv.py \
  tests/test_observabilidade_serving.py \
  tests/test_persistencia_metricas.py -q
```

**Resultado: 11 testes aprovados, 1 aviso.** O aviso vem do MLflow sobre o type hint de `predict` não ser utilizado por sua validação automática. O registro do modelo possui assinatura explícita; o aviso ainda deve ser acompanhado. Não foi executada a suíte completa nesta revisão documental.

| Área | Evidência coberta |
| --- | --- |
| Dispersão da CV | Valores conhecidos para seis métricas, folds constantes, um fold e entrada vazia |
| Integração da CV | Execução pequena com três folds e verificação de registros MLflow por mocks e métricas Prometheus |
| Serving | Média ponderada por imóvel, buckets cumulativos, localidades desconhecidas e campos inválidos |
| Adaptador HTTP | Preservação de resposta e contagem para 200/400/500; exclusão de scrape; formatos JSON |
| Persistência | Leitura do snapshot após remover o produtor; 503 sem arquivo; preservação do snapshot anterior quando a coleta falha |

O teste de integração de CV usa chamadas MLflow substituídas por mocks: não é, sozinho, prova de disponibilidade do servidor remoto ou de persistência em S3.

## Verificações operacionais anteriores nesta sessão

Foram verificados o scrape do exportador após reinício, os dados de média/desvio para três modelos, os indicadores de qualidade e holdout e os frames retornados pelo Grafana. A avaliação executada por `scripts.recalcular_metricas` concluiu as etapas 1–18 sem promoção do modelo. A correção dos painéis removeu uma união por coluna inexistente e confirmou séries nomeadas para média/desvio nas seis métricas.

Essas observações são pontuais e não garantem disponibilidade futura, todos os cenários de carga ou renderização em qualquer versão do navegador. A validação pela API do Grafana não substitui uma inspeção visual quando houver nova alteração de transformação.

## Verificações restantes

- Executar e revisar a suíte completa `tests/` no ambiente pretendido, com atenção aos testes que registram artefatos.
- Executar Ruff e análise de tipos global, registrar saídas reais e corrigir divergências antes de declarar conformidade.
- Auditar regras de uma classe por arquivo, nomes, controle de fluxo e captura de exceções.
- Validar isolamento de dados além dos asserts e o comportamento em grupos pequenos.
- Cobrir fallback por suficiência, ensemble efetivo e persistência completa do tuning quando implementados.
- Validar operação com dependências fixadas, restauração de backup e controles de acesso.

Os testes existentes em [test_pipeline_completo.py](../tests/test_pipeline_completo.py) fazem parte do projeto, mas seu nome não comprova cobertura integral de todas as 20 etapas e serviços.

## Comandos de revisão

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check app_build scripts tests
docker compose --profile servico_ml --profile dashboard --profile serving config --quiet
```

Esses comandos são instruções para verificações adicionais, não resultados certificados nesta revisão. [Pendências gerais](Final_Audit.md).

## Validação da documentação nesta revisão

Foram conferidos 28 documentos Markdown: README, guias, especificações, regras, artefatos técnicos e processo de autoria. Links internos e âncoras foram resolvidos; três blocos Python/JSON foram analisados sintaticamente; a lista de 40 campos da API foi comparada à tupla de saída do empacotador. `docker compose --profile servico_ml --profile dashboard --profile serving config --quiet` e `git diff --check` terminaram sem erros. Essas verificações não executam todos os exemplos nem garantem disponibilidade dos endereços externos.
