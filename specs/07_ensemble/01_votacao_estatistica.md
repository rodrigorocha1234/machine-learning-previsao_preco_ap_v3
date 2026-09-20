# Técnicas de Votação Baseadas em Friedman/Nemenyi

## Parâmetro obrigatório

A utilização das técnicas de votação deve ser controlada por configuração:

```yaml
ensemble:
  usar_votacao: true
```

ou:

```yaml
ensemble:
  usar_votacao: false
```

O valor padrão recomendado é:

```yaml
usar_votacao: false
```

para que ensemble seja uma decisão explícita.

## Comportamento quando `usar_votacao = false`

A pipeline deve:

1. executar normalmente Grid Search;
2. executar RepeatedKFold;
3. executar Friedman;
4. executar Nemenyi quando aplicável;
5. identificar o grupo estatisticamente elegível;
6. selecionar o melhor modelo individual dentro desse grupo;
7. não criar ensemble;
8. treinar o modelo individual campeão;
9. calcular métricas técnicas e de negócio;
10. registrar o processo no MLflow.

## Comportamento quando `usar_votacao = true`

A pipeline deve:

1. executar toda a avaliação estatística;
2. formar o grupo de modelos elegíveis;
3. construir estratégias de votação somente com modelos elegíveis;
4. validar as estratégias de ensemble usando a mesma metodologia;
5. comparar o melhor ensemble com o melhor modelo individual;
6. selecionar o candidato final sem usar o holdout para tuning.

## Grupo elegível

- executar Friedman;
- se significativo, executar Nemenyi;
- manter o melhor modelo e os modelos sem diferença significativa em relação ao melhor;
- excluir da votação modelos estatisticamente inferiores ao grupo de referência.

Se Friedman não for significativo, todos os modelos permanecem estatisticamente não diferenciados.

## Técnica A — média simples

\[
\hat y_{ens}=\frac{1}{K}\sum_{k=1}^K\hat y_k
\]

## Técnica B — média ponderada pelo erro

Peso inversamente proporcional ao RMSE validado:

\[
w_k = \frac{1/RMSE_k}{\sum_j 1/RMSE_j}
\]

\[
\hat y_{ens}=\sum_k w_k\hat y_k
\]

## Técnica C — peso pelo ranking médio

\[
w_k = \frac{1/rank_k}{\sum_j 1/rank_j}
\]

## Técnica D — stacking opcional

Somente em evolução futura e com previsões out-of-fold adequadas.

Não usar o holdout final para treinar o meta-modelo.

## Strategy

```python
class EstrategiaVotacao(ABC):
    @abstractmethod
    def combinar(
        self,
        predicoes: dict[str, np.ndarray],
        metricas: dict
    ) -> np.ndarray:
        ...
```

## Construtor

O `ConstrutorEnsemble` deve verificar a configuração antes de trabalhar:

```python
if not configuracao.ensemble.usar_votacao:
    return None
```

## MLflow

Registrar:

```text
ensemble_pkg/usado = true|false
ensemble_pkg/estrategia
ensemble_pkg/modelos_participantes
ensemble_pkg/pesos
ensemble_pkg/rmse
ensemble_pkg/mae
ensemble_pkg/r2
```

Quando `usar_votacao = false`, registrar explicitamente:

```text
ensemble_pkg/usado = false
```

para manter rastreabilidade.
