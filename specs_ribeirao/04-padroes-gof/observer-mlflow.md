# Spec — Observer para MLflow

## Objetivo
Impedir que treinamento, validação e seleção dependam diretamente de chamadas `mlflow.*`.

## Subject
```python
class TrainingSubject:
    def attach(self, observer): ...
    def detach(self, observer): ...
    def notify(self, event): ...
```

## Eventos mínimos
- `TrainingStarted`
- `FoldStarted`
- `GridSearchCompleted`
- `FoldEvaluated`
- `ModelEvaluationCompleted`
- `FriedmanCompleted`
- `NemenyiCompleted`
- `ChampionSelected`
- `FinalModelFitted`
- `ArtifactGenerated`
- `TrainingFailed`

## Observer
```python
class MLflowObserver:
    def update(self, event): ...
```

## Comportamento
### GridSearchCompleted
Logar:
- modelo;
- best params;
- best inner-CV MAE;
- duração.

### FoldEvaluated
Logar métricas usando chave com modelo/fold ou child run.

### ModelEvaluationCompleted
Logar média, mediana, desvio, IC e ranks.

### ChampionSelected
Logar nome do campeão e justificativa estruturada.

### FinalModelFitted
Logar pipeline, signature, input example e artefatos de interpretação.

## Organização recomendada de runs
- parent run: experimento completo;
- nested run por modelo;
- opcionalmente, métricas de folds como tabela/artifact para evitar centenas de runs.

## Testabilidade
Criar `InMemoryObserver` para testes unitários e `CompositeObserver` para registrar simultaneamente em MLflow + logs locais.
