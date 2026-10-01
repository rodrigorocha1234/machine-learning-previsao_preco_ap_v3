# Registro de decisões e limitações

Este registro descreve decisões efetivamente presentes. Substitui afirmações anteriores de implementação integral que não correspondiam ao código.

| Decisão | Motivo e consequência |
| --- | --- |
| Fábricas e estratégias para estimadores/tuning | Seleção por configuração, com implementações grid/random/nenhum |
| Pré-processamento no pipeline da busca | Imputação, escala e encoding ajustados dentro do treino de cada fold |
| Divisões externas compartilhadas | Comparação dos modelos sobre as mesmas partições |
| Desvio padrão com `ddof=0` | Dispersão descritiva dos scores externos; mantém a convenção anterior do RMSE |
| Observer síncrono para MLflow | Desacopla eventos de tracking; não há processamento assíncrono garantido |
| Cofre lógico em memória | Controla uma liberação por instância; não oferece criptografia e depende de asserts |
| Previsões geográficas como médias do lote | Não exige consultar uma base externa; valores mudam com os imóveis enviados |
| Média no Grafana por imóvel | Usa soma/contagem, evitando dar o mesmo peso a lotes de tamanhos diferentes |
| Aplicação nativa de scoring MLflow com adaptador ASGI | Mantém contrato e inferência MLflow, acrescentando coleta HTTP |
| Versão resolvida na inicialização | Identificação da telemetria corresponde à versão efetivamente carregada |
| Snapshot atômico de treino | Permite coleta após encerramento do processo; é estado local operacional, não substituto do histórico MLflow |
| Exportador separado, volume somente leitura | Mantém disponibilidade das métricas sem manter um treinamento aberto |
| Barras de média/desvio por modelo no Grafana | Usa diretamente labels dos frames; evita união por uma coluna `modelo` ausente |

## Escopo dos adaptadores

As condições de `AdaptadorServing` tratam protocolo ASGI, HTTP e formatos JSON. O exportador de snapshot trata disponibilidade de arquivo e rotas HTTP. Esses adapters não selecionam algoritmos de negócio. As exceções de APIs externas são tratadas localmente. Isso não certifica que todos os condicionais existentes no projeto estejam justificados ou conformes à [regra de controle de fluxo](../rules/02_no_if_rules.md).

O serving instrumentado usa um worker. Múltiplos workers exigem uma estratégia de agregação de métricas antes de mudar essa configuração. Labels geográficas são limitadas às referências conhecidas; localidades desconhecidas usam categorias fixas. Não são usados preços, endereços ou identificadores individuais como labels.

## Pendências mantidas explícitas

O comitê selecionado não se torna VotingRegressor; o fallback de suficiência não foi integrado ao caminho vetorizado; os descontos ainda são constantes no código; histórico completo do tuning não é persistido; há dependências sem lock central e instalação de pacotes na inicialização de contêineres. Consulte [o estado dos requisitos](Technical_Specification.md) antes de afirmar que uma decisão do desenho original já está implementada.
