# Spec 01 — Pipeline de machine learning

## Requisitos

Carregar/validar configuração e dados; staging; separar desenvolvimento/holdout; restringir holdout até congelar a escolha final; EDA; comparação temporal de dados; Nested CV; estatística; seleção; tuning e treino finais; avaliação única por execução; negócio; MLflow/Registry e serving.

O pré-processamento deve permanecer dentro da validação. As mesmas divisões externas devem ser utilizadas por todos os modelos; validação externa e holdout não participam do tuning.

## Implementação atual

RepeatedKFold externo 5 × 3 e KFold interno de 5 splits, conforme YAML. O split 80/20 é aleatório com semente e sem estratificação. O cofre oferece isolamento lógico por instância, não criptografia. As etapas 7 e 20 são marcadores; o bloqueio ocorre na 6, a liberação efetiva na 17 e o serving é iniciado pelo Compose.

As seis métricas são RMSE, MAE, MSE, R², RMSE relativo e MAPE. Agregações: média, mediana e desvio padrão `ddof=0` nos folds externos. O desvio mede variabilidade entre divisões, não intervalo de confiança.

Friedman utiliza RMSE; Nemenyi calcula comparações e só marca significância quando Friedman também é significativo. Shapiro e análise paramétrica são complementares. A seleção configurável não garante suporte completo a outra métrica principal em todos os módulos.

Comparação temporal real, isolamento mais robusto e ensemble efetivo permanecem pendentes. [Detalhes da CV](../docs/validacao_cruzada_e_tuning.md) e [matriz de implementação](../production_artifacts/Technical_Specification.md).
