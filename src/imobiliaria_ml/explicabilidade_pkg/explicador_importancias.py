"""Explicador de relevância de atributos para modelos não-lineares."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


class ExplicadorImportancias:
    """Calcula feature importance nativa e permutation importance para modelos complexos."""

    def calcular_importancias(
        self,
        pipeline_ajustada: Pipeline,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> pd.DataFrame:
        """Gera a tabela de importância de atributos para o modelo.

        Parameters
        ----------
        pipeline_ajustada : Pipeline
            Pipeline treinada.
        X_val : pd.DataFrame
            Conjunto de validação/holdout.
        y_val : pd.Series
            Alvo de validação/holdout.

        Returns
        -------
        pd.DataFrame
            DataFrame ordenado pela importância média da permutação.
        """
        resultado_perm = permutation_importance(
            pipeline_ajustada,
            X_val,
            y_val,
            scoring="neg_root_mean_squared_error",
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )

        nomes_features = list(X_val.columns)
        importancias_media = resultado_perm.importances_mean
        importancias_desvio = resultado_perm.importances_std

        df_importancias = pd.DataFrame(
            {
                "feature": nomes_features,
                "importancia_permutacao_media": importancias_media,
                "importancia_permutacao_desvio": importancias_desvio,
            }
        ).sort_values(by="importancia_permutacao_media", ascending=False).reset_index(drop=True)

        return df_importancias
