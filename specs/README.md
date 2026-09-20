# Spec Driven — Previsão de Preço de Imóveis

Este diretório contém as especificações modulares do projeto de Machine Learning para previsão de preço de casas/imóveis.

## Objetivos principais

- prever `Valor_da_Venda`;
- comparar múltiplos modelos de regressão;
- executar EDA completa antes do treinamento;
- tratar variáveis numéricas e categóricas;
- selecionar hiperparâmetros via `GridSearchCV`;
- validar os melhores hiperparâmetros com `RepeatedKFold`, usando **30 repetições**;
- aplicar **Friedman** somente depois da validação cruzada;
- aplicar **Nemenyi** somente quando Friedman indicar diferença estatisticamente significativa;
- escolher o modelo campeão com critérios técnicos + estatísticos;
- permitir ativar ou desativar técnicas de ensemble/votação por configuração;
- construir ensemble/votação apenas com modelos elegíveis pelos testes estatísticos quando a funcionalidade estiver ativada;
- calcular métricas de negócio da imobiliária com o modelo campeão;
- registrar experimentos, Grid Search, validação, gráficos, coeficientes, drift e modelos diretamente no MLflow, sem persistência local permanente de artefatos;
- servir o modelo via **MLflow Model Serving**, usando o MLflow existente no `docker-compose.yml` da raiz;
- não usar FastAPI da aplicação;
- aceitar `Percentual_Desconto` na requisição de inferência;
- permitir ingestão de novos dados e retreinamento controlado;
- permitir futura substituição do backend local por processamento distribuído;
- usar padrões GoF e **um módulo Python por classe**;
- classes, métodos e atributos em português.

## Variáveis iniciais

| Variável | Tipo | Papel |
|---|---|---|
| `Zona` | categórica | localização/região do imóvel |
| `Quartos` | numérica discreta | característica |
| `Banheiros` | numérica discreta | característica |
| `Vagas` | numérica discreta | característica |
| `Metragem` | numérica contínua | característica |
| `Valor_da_Venda` | numérica contínua | alvo |

Variáveis futuras recomendadas: `Idade_Imovel`, `Tipo_Imovel`, `Condominio`, `IPTU`, `Distancia_Centro`, `Latitude`, `Longitude`, `Padrao_Acabamento`, `Possui_Elevador`, `Andar`, `Data_Venda`.

## Fluxo oficial

```text
Dados
  ↓
Validação de esquema e qualidade
  ↓
EDA
  ↓
Separação treino/holdout final
  ↓
Pipeline de pré-processamento
  ↓
GridSearchCV por modelo
  ↓
Melhores hiperparâmetros de cada modelo
  ↓
RepeatedKFold (30 repetições)
  ↓
Métricas por repetição e modelo
  ↓
Friedman
  ↓ se p < alpha
Nemenyi
  ↓
Grupo estatisticamente elegível
  ↓
Parâmetro `usar_votacao`
  ├── false → selecionar melhor modelo individual elegível
  └── true  → avaliar estratégias de votação com modelos elegíveis
                    ↓
              selecionar melhor candidato
  ↓
Treino final
  ↓
Holdout final
  ↓
Métricas técnicas + métricas imobiliárias
  ↓
Registro/Registry MLflow
  ↓
MLflow Model Serving
  ↓
Monitoramento de drift
  ↓
Retreinamento controlado
```

## Regra estatística importante

Os resultados do Grid Search **não entram no Friedman/Nemenyi**. O Grid Search existe apenas para escolher hiperparâmetros. O Friedman recebe os resultados da validação cruzada posterior.

Para evitar tratar os folds internos como observações independentes, cada repetição do `RepeatedKFold` gera um valor agregado por modelo (média dos folds daquela repetição). Assim, a matriz estatística possui aproximadamente:

```text
linhas = 30 repetições
colunas = modelos
valor = métrica média da repetição
```

Todos os modelos devem usar exatamente as mesmas partições.

## Diretórios

Consulte `00_visao_geral/02_estrutura_diretorios.md`.


## Atualização de ensembles concretos

As implementações de ensemble devem usar diretamente:

- `sklearn.ensemble.VotingRegressor`
- `sklearn.ensemble.StackingRegressor`
- `sklearn.ensemble.BaggingRegressor`

O catálogo de candidatos também inclui:

- `sklearn.ensemble.GradientBoostingRegressor`
- `catboost.CatBoostRegressor`

Voting e Stacking utilizam somente modelos estatisticamente elegíveis após Friedman/Nemenyi.

## Convenção de packages Python

Todos os diretórios Python dentro de `src/imobiliaria_ml/` devem usar o sufixo `_pkg`.

Exemplos: `modelos_pkg`, `ensemble_pkg`, `validacao_pkg`, `mlflow_pkg`.

A regra `1 classe = 1 arquivo .py` permanece obrigatória.

## Tipagem sem `Any`

A implementação não deve utilizar `typing.Any` nem `Any`.

Use tipos concretos, unions específicas, `TypedDict`, `Protocol`, `TypeVar`, `Generic` ou `dataclass` para manter contratos explícitos.


## Enums de domínio

A implementação usa `StrEnum` para categorias internas fechadas, incluindo modelos, ensembles, scalers, eventos do Observer, nível de drift, origem dos pesos, carregadores, métricas e estimadores finais do stacking.

Isso reduz strings mágicas e reforça a tipagem sem `Any`.


## Generic e TypeVar

A implementação usa `Generic` e `TypeVar` quando precisa preservar tipos entre entrada e saída, especialmente em Strategies, Executors, Adapters e objetos de resultado.

Quando o contrato for comportamental, prefira `Protocol` como bound.

Não usar generics apenas para esconder tipagem indefinida.
