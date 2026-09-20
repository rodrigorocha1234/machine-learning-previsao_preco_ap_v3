# Spec — Critérios de Aceite

## Dados
- [ ] colunas obrigatórias validadas;
- [ ] tipos e limites validados;
- [ ] nulos/duplicatas reportados.

## Modelagem
- [ ] pelo menos 10 Strategies executadas;
- [ ] Grid Search interno por fold externo;
- [ ] 30 folds externos exatamente;
- [ ] nenhuma transformação ajustada no dataset completo antes do CV;
- [ ] métricas por fold persistidas.

## Estatística
- [ ] Friedman usa scores pareados dos mesmos 30 folds;
- [ ] Nemenyi só é usado como pós-hoc quando Friedman é significativo;
- [ ] ranks médios persistidos;
- [ ] regra de desempate documentada.

## Equações/interpretação
- [ ] equação expandida para modelos lineares;
- [ ] função matemática correta para modelos não lineares;
- [ ] hiperparâmetros finais de todos os modelos;
- [ ] parâmetros aprendidos quando disponíveis;
- [ ] interpretação de negócio e limitações.

## MLflow
- [ ] logging conectado por Observer;
- [ ] champion registrado;
- [ ] model signature registrada;
- [ ] input example registrado;
- [ ] model serving funciona em `/invocations`;
- [ ] não existe API FastAPI própria.

## Negócio
- [ ] resposta contém 30 métricas;
- [ ] preço informado não é usado como feature do estimador;
- [ ] score de confiança é identificado como índice operacional;
- [ ] comparáveis usam apenas base histórica disponível no artefato.

## Relatórios
- [ ] relatório EDA gerado;
- [ ] relatório de modelos gerado;
- [ ] artefatos do Friedman/Nemenyi anexados;
- [ ] versão de bibliotecas/seed/dataset fingerprint registrados.
