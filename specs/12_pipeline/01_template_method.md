# Pipeline como Template Method

Classe base:

```python
class PipelineTreinamento(ABC):

    def executar(self):
        dados = self.carregar_dados()
        self.validar_dados(dados)
        self.executar_eda(dados)

        treino, holdout = self.separar_holdout(dados)

        candidatos = self.criar_candidatos()

        melhores = self.executar_grid_search(
            candidatos,
            treino
        )

        resultados = self.validar_repeated_kfold(
            melhores,
            treino
        )

        estatistica = self.executar_testes_estatisticos(
            resultados
        )

        melhor_individual = self.selecionar_modelo_individual(
            estatistica
        )

        candidato_final = melhor_individual

        if self.configuracao.ensemble.usar_votacao:
            ensemble = self.avaliar_ensembles(
                estatistica,
                treino
            )

            candidato_final = self.comparar_candidatos(
                melhor_individual,
                ensemble
            )

        modelo_final = self.treinar_final(
            candidato_final,
            treino
        )

        avaliacao = self.avaliar_holdout(
            modelo_final,
            holdout
        )

        self.gerar_curva_aprendizado(
            modelo_final,
            treino
        )

        self.calcular_metricas_negocio(
            modelo_final,
            holdout
        )

        self.registrar_modelo(
            modelo_final,
            avaliacao
        )
```

## Regra

O parâmetro `usar_votacao` altera apenas a etapa posterior aos testes estatísticos.

Grid Search, RepeatedKFold, Friedman e Nemenyi continuam sendo executados normalmente.

## MLflow

A pipeline deve notificar objetos em memória.

Exemplo:

```python
self.notificar(
    "validacao_finalizada",
    {
        "resultados": dataframe_resultados,
        "resumo": resumo_metricas
    }
)
```

O MLflow é responsável pela persistência final desses artefatos.


## Resultados tipados

Quando a pipeline trafegar objetos que encapsulam modelos, usar dataclasses genéricas.

Exemplo:

```python
@dataclass(frozen=True)
class ResultadoTreinamento(Generic[TModelo]):
    modelo: TModelo
    metricas: MetricasRegressao
```

Isso preserva o tipo concreto do modelo ao longo do fluxo.
