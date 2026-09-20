# Parâmetro `usar_votacao`

## Objetivo

Permitir que o mesmo projeto opere em dois modos sem alterar código.

## Modo individual

```yaml
ensemble:
  usar_votacao: false
```

Fluxo:

```text
Friedman/Nemenyi
      ↓
grupo elegível
      ↓
seleção do melhor modelo individual
      ↓
campeão
```

## Modo ensemble

```yaml
ensemble:
  usar_votacao: true
```

Fluxo:

```text
Friedman/Nemenyi
      ↓
grupo elegível
      ↓
melhor individual
      +
estratégias de votação
      ↓
comparação validada
      ↓
campeão
```

## Origem da configuração

A origem oficial é o arquivo de configuração.

Opcionalmente, a CLI pode sobrescrever:

```bash
python -m imobiliaria_ml.aplicacao_pkg.treinar --usar-votacao
```

ou:

```bash
python -m imobiliaria_ml.aplicacao_pkg.treinar --nao-usar-votacao
```

A aplicação deve registrar no MLflow o valor efetivamente utilizado.

## Regra

O parâmetro não pode alterar:

- folds;
- Grid Search;
- dados usados;
- Friedman;
- Nemenyi.

Ele controla somente a avaliação de ensembles após a etapa estatística.
