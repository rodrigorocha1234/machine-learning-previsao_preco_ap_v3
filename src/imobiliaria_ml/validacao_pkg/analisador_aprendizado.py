"""Analisador de curva de aprendizado para diagnóstico de underfitting e overfitting."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline


class AnalisadorAprendizado:
    """Gera dados e figuras da curva de aprendizado usando learning_curve."""

    def gerar_curva(
        self,
        modelo_pipeline: Pipeline,
        X_treino: pd.DataFrame,
        y_treino: pd.Series,
        cv: int = 5,
    ) -> tuple[Figure, pd.DataFrame, dict[str, str | float]]:
        """Calcula a curva de aprendizado e produz a figura e diagnóstico em memória.

        Parameters
        ----------
        modelo_pipeline : Pipeline
            Pipeline treinada ou configurada.
        X_treino : pd.DataFrame
            Conjunto de características.
        y_treino : pd.Series
            Conjunto do alvo.
        cv : int
            Número de divisões de validação.

        Returns
        -------
        tuple[Figure, pd.DataFrame, dict[str, str | float]]
            Figura gráfica, DataFrame com dados dos pontos e diagnóstico textual.
        """
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=modelo_pipeline,
            X=X_treino,
            y=y_treino,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            train_sizes=np.linspace(0.2, 1.0, 5),
            n_jobs=-1,
        )

        train_rmse = -np.mean(train_scores, axis=1)
        test_rmse = -np.mean(test_scores, axis=1)

        df_curva = pd.DataFrame(
            {
                "tamanho_treino": train_sizes,
                "rmse_treino": train_rmse,
                "rmse_validacao": test_rmse,
            }
        )

        gap_final = float(test_rmse[-1] - train_rmse[-1])
        if gap_final > (0.3 * test_rmse[-1]):
            diagnostico = "Possível overfitting: gap considerável entre treino e validação."
        elif test_rmse[-1] > (test_rmse[0] * 0.95):
            diagnostico = "Possível underfitting: erro alto persistente em validação."
        else:
            diagnostico = "Boa capacidade de generalização com convergência estável."

        info_diagnostico: dict[str, str | float] = {
            "diagnostico": diagnostico,
            "gap_final_rmse": gap_final,
            "rmse_treino_final": float(train_rmse[-1]),
            "rmse_validacao_final": float(test_rmse[-1]),
        }

        # Criação da figura gráfica em memória
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_sizes, train_rmse, "o-", color="#1f77b4", label="Treino (RMSE)")
        ax.plot(train_sizes, test_rmse, "o-", color="#ff7f0e", label="Validação (RMSE)")
        ax.set_title("Curva de Aprendizado (Diagnóstico Under/Overfitting)")
        ax.set_xlabel("Exemplos de Treinamento")
        ax.set_ylabel("RMSE")
        ax.legend(loc="best")
        ax.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()

        return fig, df_curva, info_diagnostico
