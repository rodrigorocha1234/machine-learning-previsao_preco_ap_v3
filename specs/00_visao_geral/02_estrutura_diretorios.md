# Estrutura de Diretórios

```text
projeto/
├── docker-compose.yml                # já existente na raiz
├── pyproject.toml
├── .env
├── dados/
│   ├── bruto/
│   ├── processado/
│   ├── referencia_drift/
│   └── novos/
├── src/
│   └── imobiliaria_ml/
│       ├── enums_pkg/
│       │   ├── tipo_modelo.py
│       │   ├── tipo_ensemble.py
│       │   ├── tipo_escalonador.py
│       │   ├── tipo_evento.py
│       │   ├── nivel_drift.py
│       │   ├── origem_peso_voting.py
│       │   ├── tipo_carregador.py
│       │   ├── tipo_metrica.py
│       │   └── tipo_estimador_final.py
│       ├── carregamento_pkg/
│       │   ├── carregador_base.py
│       │   ├── carregador_csv.py
│       │   ├── carregador_excel.py
│       │   ├── carregador_banco.py
│       │   └── fabrica_carregadores.py
│       ├── qualidade_pkg/
│       │   ├── validador_esquema.py
│       │   └── validador_qualidade.py
│       ├── eda_pkg/
│       │   ├── analisador_descritivo.py
│       │   ├── analisador_distribuicao.py
│       │   └── gerador_relatorio_eda.py
│       ├── preprocessamento_pkg/
│       │   ├── pre_processador.py
│       │   ├── estrategia_escalonamento.py
│       │   ├── escalonador_standard.py
│       │   ├── escalonador_minmax.py
│       │   ├── escalonador_robusto.py
│       │   └── tratador_categorico.py
│       ├── modelos_pkg/
│       │   ├── estrategia_modelo.py
│       │   ├── regressao_linear.py
│       │   ├── regressao_multipla.py
│       │   ├── regressao_polinomial.py
│       │   ├── regressao_ridge.py
│       │   ├── regressao_lasso.py
│       │   ├── regressao_elastic_net.py
│       │   ├── regressao_arvore.py
│       │   ├── regressao_random_forest.py
│       │   ├── regressao_gradient_boosting.py
│       │   ├── regressao_svr.py
│       │   ├── regressao_rede_neural.py
│       │   ├── regressao_xgboost.py
│       │   ├── regressao_lightgbm.py
│       │   ├── regressao_catboost.py
│       │   └── fabrica_modelos.py
│       ├── selecao_pkg/
│       │   ├── seletor_hiperparametros.py
│       │   └── catalogo_grades.py
│       ├── validacao_pkg/
│       │   ├── validador_repeated_kfold.py
│       │   ├── calculador_metricas.py
│       │   ├── teste_friedman.py
│       │   ├── teste_nemenyi.py
│       │   ├── seletor_campeao.py
│       │   └── analisador_aprendizado.py
│       ├── ensemble_pkg/
│       │   ├── estrategia_ensemble.py
│       │   ├── votacao_regressor.py
│       │   ├── votacao_regressor_ponderada.py
│       │   ├── stacking_regressor_imobiliario.py
│       │   ├── bagging_regressor_imobiliario.py
│       │   ├── calculador_pesos_rmse.py
│       │   ├── calculador_pesos_ranking.py
│       │   ├── seletor_modelos_ensemble.py
│       │   └── fabrica_ensemble.py
│       ├── explicabilidade_pkg/
│       │   ├── extrator_coeficientes.py
│       │   ├── interpretador_modelo.py
│       │   └── explicador_importancias.py
│       ├── negocio_pkg/
│       │   ├── calculador_desconto_seguro.py
│       │   ├── calculador_metricas_imobiliaria.py
│       │   └── regras_imobiliaria.py
│       ├── drift_pkg/
│       │   ├── detector_drift_numerico.py
│       │   ├── detector_drift_categorico.py
│       │   ├── detector_drift_predicao.py
│       │   └── monitor_drift.py
│       ├── mlflow_pkg/
│       │   ├── observador.py
│       │   ├── sujeito_observavel.py
│       │   ├── observador_mlflow.py
│       │   ├── registrador_modelo.py
│       │   └── modelo_imobiliario_pyfunc.py
│       ├── distribuido_pkg/
│       │   ├── executor_ml.py
│       │   ├── executor_sklearn.py
│       │   └── adaptador_distribuido.py
│       ├── pipeline_pkg/
│       │   └── pipeline_treinamento.py
│       └── aplicacao_pkg/
│           ├── treinar.py
│           ├── avaliar_drift.py
│           └── promover_modelo.py
├── tests/
└── specs/
    └── ...
```

Regra: **uma classe por módulo Python**.
