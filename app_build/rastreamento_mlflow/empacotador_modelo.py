from typing import Final, Protocol, override, runtime_checkable

import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel, PythonModelContext
from sklearn.base import BaseEstimator

from app_build.regras_negocio.motor_imobiliario import MotorImobiliario


@runtime_checkable
class ProtocoloEnriquecedor(Protocol):
    def enriquecer_dataframe(
        self, dados_x: pd.DataFrame, vetor_previsoes: pd.Series
    ) -> pd.DataFrame: ...


class EnriquecedorGenerico:
    def enriquecer_dataframe(
        self, dados_x: pd.DataFrame, vetor_previsoes: pd.Series
    ) -> pd.DataFrame:
        metragens = pd.Series(
            np.asarray(dados_x.get("Metragem", 1.0), dtype=np.float64),
            index=dados_x.index,
        ).clip(lower=1.0)
        previsoes_m2 = vetor_previsoes / metragens
        return dados_x.assign(
            valor_previsto=vetor_previsoes,
            valor_m2_previsto=previsoes_m2,
            # --- Nível Global ---
            indice_imovel_global=1.0,
            diferenca_perc_global=0.0,
            global_desconto_5=vetor_previsoes * 0.95,
            global_desconto_10=vetor_previsoes * 0.90,
            global_desconto_15=vetor_previsoes * 0.85,
            global_faixa_segura_piso=vetor_previsoes * 0.90,
            global_faixa_segura_teto=vetor_previsoes,
            # --- Nível Zona ---
            indice_imovel_zona=1.0,
            diferenca_perc_zona=0.0,
            zona_desconto_5=vetor_previsoes * 0.95,
            zona_desconto_10=vetor_previsoes * 0.90,
            zona_desconto_15=vetor_previsoes * 0.85,
            zona_faixa_segura_piso=vetor_previsoes * 0.90,
            zona_faixa_segura_teto=vetor_previsoes,
            # --- Nível Bairro ---
            indice_imovel_bairro=1.0,
            diferenca_perc_bairro=0.0,
            bairro_desconto_5=vetor_previsoes * 0.95,
            bairro_desconto_10=vetor_previsoes * 0.90,
            bairro_desconto_15=vetor_previsoes * 0.85,
            bairro_faixa_segura_piso=vetor_previsoes * 0.90,
            bairro_faixa_segura_teto=vetor_previsoes,
        )


class EmpacotadorModelo(PythonModel):
    def __init__(
        self,
        pipeline_scikit: BaseEstimator,
        motor_imobiliario: MotorImobiliario | None = None,
    ) -> None:
        self.pipeline_scikit: Final[BaseEstimator] = pipeline_scikit
        self.motor_imobiliario: Final[ProtocoloEnriquecedor] = (
            motor_imobiliario or EnriquecedorGenerico()
        )

    @override
    def predict(
        self,
        context: PythonModelContext | None,
        model_input: pd.DataFrame | np.ndarray,
        params: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        df_entrada = pd.DataFrame(model_input)
        colunas_indesejadas: tuple[str, ...] = (
            "valor_m2",
            "media_valor_m2_bairro",
            "media_valor_m2_zona",
        )
        cols_entrada = set(df_entrada.columns)
        cols_drop: tuple[str, ...] = tuple(
            filter(cols_entrada.__contains__, colunas_indesejadas)
        )
        df_limpo = df_entrada.drop(columns=list(cols_drop))

        previsoes = self.pipeline_scikit.predict(df_limpo)
        vetor_prev = pd.Series(previsoes, index=df_entrada.index)

        df_enriquecido = self.motor_imobiliario.enriquecer_dataframe(
            df_entrada, vetor_prev
        )
        # Colunas de saída organizadas por hierarquia: GLOBAL → ZONA → BAIRRO
        colunas_resultado: tuple[str, ...] = (
            # Previsão base
            "valor_previsto",
            "valor_m2_previsto",
            # Nível Global
            "indice_imovel_global",
            "diferenca_perc_global",
            "global_desconto_5",
            "global_desconto_10",
            "global_desconto_15",
            "global_faixa_segura_piso",
            "global_faixa_segura_teto",
            # Nível Zona
            "indice_imovel_zona",
            "diferenca_perc_zona",
            "zona_desconto_5",
            "zona_desconto_10",
            "zona_desconto_15",
            "zona_faixa_segura_piso",
            "zona_faixa_segura_teto",
            # Nível Bairro
            "indice_imovel_bairro",
            "diferenca_perc_bairro",
            "bairro_desconto_5",
            "bairro_desconto_10",
            "bairro_desconto_15",
            "bairro_faixa_segura_piso",
            "bairro_faixa_segura_teto",
        )
        return df_enriquecido[list(colunas_resultado)]
