# Padrões GoF

| Padrão | Uso |
|---|---|
| Strategy | algoritmos de regressão, escalonadores e estratégias de votação |
| Factory Method / Abstract Factory | criação de carregadores e modelos por configuração |
| Template Method | fluxo completo de treinamento |
| Observer | publicação de eventos para MLflow |
| Adapter | backend local hoje e backend distribuído futuramente |
| Chain of Responsibility | validações de dados/qualidade antes do treino |
| Facade | opcional para simplificar execução externa da pipeline |

## Observer + MLflow

O treinamento não deve chamar MLflow em toda classe. As classes publicam eventos:

```python
sujeito.notificar("gridsearch_finalizado", dados_evento)
sujeito.notificar("validacao_finalizada", dados_evento)
sujeito.notificar("modelo_campeao", dados_evento)
```

`ObservadorMLflow` traduz os eventos em:

- `mlflow.log_params`
- `mlflow.log_metric`
- `mlflow.log_metrics`
- `mlflow.log_artifact`
- `mlflow.log_table`
- `mlflow.pyfunc.log_model`
- registro/promoção no Model Registry


## Enums integrados aos padrões

Factories devem receber enums tipados como `TipoModelo`, `TipoCarregador` e `TipoEnsemble`.

O Observer deve receber `TipoEvento` em vez de strings livres.

Strategies de scaler devem usar `TipoEscalonador`.

A seleção de pesos do Voting deve usar `OrigemPesoVoting`.


## Tipagem dos padrões

Strategies, Executors e Adapters devem usar `Generic` e `TypeVar` quando precisarem preservar o tipo concreto.

Preferir `Protocol` como bound para contratos comportamentais.
