"""Executor de busca e otimização de hiperparâmetros com GridSearchCV."""

import time
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from ..enums_pkg.tipo_modelo import TipoModelo
from ..modelos_pkg.regressor_protocol import RegressorProtocol
from ..modelos_pkg.estrategia_modelo import EstrategiaModelo
from .catalogo_grades import CatalogoGrades
from .resultado_grid_search import ResultadoGridSearch


class SeletorHiperparametros:
    """Executa a seleção de melhores hiperparâmetros utilizando o conjunto de desenvolvimento."""

    def __init__(self, cv_splits: int = 5, n_jobs: int = -1) -> None:
        self._cv_splits = cv_splits
        self._n_jobs = n_jobs
        self._catalogo_grades = CatalogoGrades()

    def otimizar(
        self,
        estrategia: EstrategiaModelo[RegressorProtocol],
        pipeline_base: Pipeline,
        X_treino: pd.DataFrame,
        y_treino: pd.Series,
        grade_customizada: dict[str, list[int | float | str | bool | tuple[int, ...] | None]] | None = None,
    ) -> ResultadoGridSearch:
        """Executa o GridSearchCV sobre a pipeline fornecida.

        Parameters
        ----------
        estrategia : EstrategiaModelo[RegressorProtocol]
            Estratégia do modelo candidato.
        pipeline_base : Pipeline
            Pipeline combinando o pré-processador e o estimador com nome 'modelo'.
        X_treino : pd.DataFrame
            Conjunto de desenvolvimento (características).
        y_treino : pd.Series
            Conjunto de desenvolvimento (alvo).
        grade_customizada : dict | None
            Grade alternativa opcional. Se None, usa a grade padrão do catálogo.

        Returns
        -------
        ResultadoGridSearch
            Resultado tipado contendo melhores parâmetros e métricas.
        """
        tipo = estrategia.obter_tipo_modelo()
        grade = grade_customizada or self._catalogo_grades.obter_grade(tipo)

        inicio = time.time()
        busca = GridSearchCV(
            estimator=pipeline_base,
            param_grid=grade,
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
            refit="rmse",
            cv=self._cv_splits,
            n_jobs=self._n_jobs,
            return_train_score=True,
        )

        busca.fit(X_treino, y_treino)
        tempo_total = float(time.time() - inicio)

        df_cv_results = pd.DataFrame(busca.cv_results_)
        melhor_score_rmse = float(-busca.best_score_)

        melhores_params: dict[str, int | float | str | bool | tuple[int, ...] | None] = {}
        for k, v in busca.best_params_.items():
            melhores_params[str(k)] = v

        return ResultadoGridSearch(
            tipo_modelo=tipo,
            melhores_parametros=melhores_params,
            melhor_score_rmse=melhor_score_rmse,
            tabela_cv_results=df_cv_results,
            tempo_total_segundos=tempo_total,
        )
