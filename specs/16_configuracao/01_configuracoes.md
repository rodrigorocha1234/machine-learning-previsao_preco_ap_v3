# Configurações

Exemplo:

```yaml
projeto:
  seed: 42
  alvo: Valor_da_Venda

validacao:
  n_splits: 5
  n_repeats: 30
  alpha_friedman: 0.05
  alpha_nemenyi: 0.05
  metrica_primaria: rmse

ensemble:
  usar_votacao: false

  tecnicas_habilitadas:
    - voting_media
    - voting_ponderado_rmse
    - voting_ponderado_ranking
    - stacking
    - bagging

  stacking:
    estimador_final: ridge

  bagging:
    estimador_base: arvore_decisao
    n_estimators: 100
    max_samples: 0.8
    bootstrap: true

mlflow:
  tracking_uri_env: MLFLOW_TRACKING_URI
  experimento: previsao-preco-imoveis
  nome_modelo: preco-imoveis
  persistencia_local_artefatos: false

negocio:
  desconto_minimo: 0
  desconto_maximo_automatico: 5
  desconto_maximo_com_aprovacao: 10

drift:
  psi_atencao: 0.10
  psi_forte: 0.25
```

## `ensemble.usar_votacao`

`false`:
- não cria Voting nem Stacking;
- seleciona melhor modelo individual elegível;
- Bagging só é avaliado se explicitamente tratado como modelo candidato.

`true`:
- permite VotingRegressor e StackingRegressor;
- usa apenas modelos estatisticamente elegíveis;
- compara ensemble contra melhor modelo individual.

## `mlflow.persistencia_local_artefatos`

Deve permanecer `false` no fluxo padrão.

Arquivos temporários somente quando alguma biblioteca exigir caminho físico.


## Enums

Após carregar o YAML, converter valores categóricos para enums de domínio.

Exemplos:

```python
tipo_modelo = TipoModelo(configuracao["modelo"])
tipo_ensemble = TipoEnsemble(valor)
tipo_escalonador = TipoEscalonador(valor)
origem_peso = OrigemPesoVoting(valor)
tipo_estimador_final = TipoEstimadorFinal(valor)
```

A configuração externa continua usando strings serializáveis, mas a camada interna da aplicação deve trabalhar com enums.
