# Spec — Padrões GoF

## Strategy — Modelos
`RegressionStrategy` encapsula criação do pipeline, grid e metadados de interpretabilidade.

## Factory Method — ModelFactory
```python
class ModelFactory:
    def create(self, model_name: str) -> RegressionStrategy: ...
```
A Factory não treina modelos; apenas devolve a Strategy correta.

## Template Method — TrainingWorkflow
Fluxo fixo:
1. validar dados;
2. criar folds;
3. tune;
4. avaliar;
5. notificar observers;
6. comparar modelos;
7. refit campeão;
8. registrar.

Subclasses podem customizar hooks, não a ordem crítica.

## Observer — MLflow
O `TrainingSubject` publica eventos. `MLflowObserver` registra métricas/parâmetros/artefatos sem acoplar classes de treinamento diretamente ao MLflow.

## Facade — RealEstateMLFacade
Métodos conceituais:
```python
fit(data)
predict(features, asking_price)
generate_eda(data)
generate_model_report()
```

## Adapter — MLflowPyFuncAdapter
Adapta o domínio para `mlflow.pyfunc.PythonModel` e transforma a saída em dataframe serializável pelo MLflow Model Serving.

## Builder — BusinessMetricsBuilder
Constrói as 30 métricas de negócio incrementalmente a partir de:
- features do imóvel;
- preço informado;
- previsão do campeão;
- estatísticas de treino;
- intervalo/calibração;
- vizinhos/comparáveis;
- metadados dos modelos.
