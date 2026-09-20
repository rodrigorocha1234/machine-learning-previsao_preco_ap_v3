"""Catálogo de grades iniciais de hiperparâmetros por TipoModelo."""

from ..enums_pkg.tipo_modelo import TipoModelo


class CatalogoGrades:
    """Provedor centralizado de grades de hiperparâmetros para GridSearchCV."""

    def obter_grade(
        self, tipo: TipoModelo
    ) -> dict[str, list[int | float | str | bool | tuple[int, ...] | None]]:
        """Retorna a grade de hiperparâmetros com prefixo 'modelo__' para a pipeline.

        Parameters
        ----------
        tipo : TipoModelo
            Tipo do modelo de regressão.

        Returns
        -------
        dict[str, list[int | float | str | bool | tuple[int, ...] | None]]
            Grade formatada para o GridSearchCV.
        """
        match tipo:
            case TipoModelo.REGRESSAO_LINEAR | TipoModelo.REGRESSAO_MULTIPLA:
                return {"modelo__fit_intercept": [True, False]}
            case TipoModelo.REGRESSAO_POLINOMIAL:
                return {
                    "modelo__poly__degree": [2, 3],
                    "modelo__linear__fit_intercept": [True, False],
                }
            case TipoModelo.RIDGE:
                return {"modelo__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
            case TipoModelo.LASSO:
                return {"modelo__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]}
            case TipoModelo.ELASTIC_NET:
                return {
                    "modelo__alpha": [0.001, 0.01, 0.1, 1.0],
                    "modelo__l1_ratio": [0.1, 0.25, 0.5, 0.75, 0.9],
                }
            case TipoModelo.ARVORE_DECISAO:
                return {
                    "modelo__max_depth": [3, 5, 8, 12, None],
                    "modelo__min_samples_split": [2, 5, 10, 20],
                    "modelo__min_samples_leaf": [1, 2, 5, 10],
                }
            case TipoModelo.RANDOM_FOREST:
                return {
                    "modelo__n_estimators": [200, 500],
                    "modelo__max_depth": [8, 12, 20, None],
                    "modelo__min_samples_leaf": [1, 2, 5],
                    "modelo__max_features": ["sqrt", 0.7, 1.0],
                }
            case TipoModelo.GRADIENT_BOOSTING:
                return {
                    "modelo__n_estimators": [100, 200, 500],
                    "modelo__learning_rate": [0.01, 0.05, 0.1],
                    "modelo__max_depth": [2, 3, 5],
                    "modelo__min_samples_leaf": [1, 2, 5],
                    "modelo__subsample": [0.7, 0.9, 1.0],
                }
            case TipoModelo.CATBOOST:
                return {
                    "modelo__iterations": [200, 500],
                    "modelo__depth": [4, 6, 8],
                    "modelo__learning_rate": [0.01, 0.05, 0.1],
                    "modelo__l2_leaf_reg": [1.0, 3.0, 5.0, 10.0],
                    "modelo__verbose": [False],
                }
            case TipoModelo.SVR:
                return {
                    "modelo__kernel": ["rbf", "linear"],
                    "modelo__C": [1.0, 10.0, 100.0],
                    "modelo__epsilon": [0.01, 0.1, 0.5],
                    "modelo__gamma": ["scale", "auto"],
                }
            case TipoModelo.REDE_NEURAL:
                return {
                    "modelo__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
                    "modelo__alpha": [0.0001, 0.001, 0.01],
                    "modelo__learning_rate_init": [0.0005, 0.001, 0.01],
                }
            case TipoModelo.XGBOOST:
                return {
                    "modelo__n_estimators": [200, 500],
                    "modelo__max_depth": [3, 5, 8],
                    "modelo__learning_rate": [0.01, 0.05, 0.1],
                    "modelo__subsample": [0.7, 0.9, 1.0],
                    "modelo__colsample_bytree": [0.7, 0.9, 1.0],
                }
            case TipoModelo.LIGHTGBM:
                return {
                    "modelo__n_estimators": [200, 500],
                    "modelo__num_leaves": [15, 31, 63],
                    "modelo__learning_rate": [0.01, 0.05, 0.1],
                    "modelo__min_child_samples": [10, 20, 40],
                    "modelo__subsample": [0.7, 0.9, 1.0],
                }
