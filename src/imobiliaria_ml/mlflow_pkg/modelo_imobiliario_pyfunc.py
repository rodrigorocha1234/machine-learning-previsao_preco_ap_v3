"""Modelo customizado PyFunc para previsão de preço imobiliário com aplicação de desconto."""

import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class ModeloImobiliarioPyFunc(mlflow.pyfunc.PythonModel):
    """Encapsulamento PyFunc que aceita features imobiliárias e Percentual_Desconto."""

    def __init__(
        self,
        modelo_pipeline: Pipeline | None = None,
        desconto_maximo_permitido: float = 10.0,
    ) -> None:
        self._modelo_pipeline = modelo_pipeline
        self._desconto_maximo = desconto_maximo_permitido

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Carrega a pipeline caso o modelo esteja sendo deserializado de artefato."""
        if hasattr(context, "artifacts") and "pipeline" in context.artifacts:
            import joblib
            self._modelo_pipeline = joblib.load(context.artifacts["pipeline"])

    def _validar_desconto(self, desconto: pd.Series | np.ndarray) -> None:
        """Verifica se os descontos informados respeitam os limites comerciais."""
        arr = np.asarray(desconto, dtype=float)
        if (arr < 0.0).any():
            raise ValueError("Percentual_Desconto não pode ser negativo.")
        if (arr > self._desconto_maximo).any():
            raise ValueError(
                f"Percentual_Desconto superior ao teto configurado de {self._desconto_maximo}%."
            )

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext | None,
        model_input: pd.DataFrame,
    ) -> pd.DataFrame:
        """Executa a inferência e calcula o preço líquido após desconto.

        Parameters
        ----------
        context : mlflow.pyfunc.PythonModelContext | None
            Contexto fornecido pelo runtime do MLflow.
        model_input : pd.DataFrame
            DataFrame contendo as features e opcionalmente Percentual_Desconto.

        Returns
        -------
        pd.DataFrame
            DataFrame com Valor_Previsto, Percentual_Desconto e Valor_Com_Desconto.
        """
        if self._modelo_pipeline is None:
            raise RuntimeError("Pipeline de modelo não carregada no objeto PyFunc.")

        dados = model_input.copy()

        # Extração e validação do desconto
        if "Percentual_Desconto" in dados.columns:
            desconto = dados.pop("Percentual_Desconto").astype(float)
        else:
            desconto = pd.Series(0.0, index=dados.index)

        self._validar_desconto(desconto)

        # Previsão das features restantes (Zona, Quartos, Banheiros, Vagas, Metragem)
        previsoes = self._modelo_pipeline.predict(dados)
        valor_previsto = np.asarray(previsoes, dtype=float)

        fator_desconto = 1.0 - (desconto.values / 100.0)
        valor_com_desconto = valor_previsto * fator_desconto

        return pd.DataFrame(
            {
                "Valor_Previsto": valor_previsto,
                "Percentual_Desconto": desconto.values,
                "Valor_Com_Desconto": valor_com_desconto,
            },
            index=dados.index,
        )
