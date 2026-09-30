from typing import Final

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from app_build.processamento_dados.extrator_atributos import ExtratorAtributos


class ConstrutorPipeline:
    def __init__(self) -> None:
        self._colunas_numericas: Final[tuple[str, ...]] = (
            "Metragem",
            "Quartos",
            "Banheiros",
            "Vagas_Garagem",
            "razao_banheiros_quartos",
            "metragem_por_quarto",
            "total_comodos",
        )
        self._colunas_categoricas: Final[tuple[str, ...]] = (
            "Zona",
            "Bairro",
        )

    def criar_preprocessador(self) -> Pipeline:
        pipeline_numerico = Pipeline(
            steps=[
                ("imputador", SimpleImputer(strategy="median")),
                ("escalador", RobustScaler()),
            ]
        )

        pipeline_categorico = Pipeline(
            steps=[
                ("imputador", SimpleImputer(strategy="most_frequent")),
                (
                    "codificador",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        transformador_colunas = ColumnTransformer(
            transformers=[
                ("numerico", pipeline_numerico, list(self._colunas_numericas)),
                ("categorico", pipeline_categorico, list(self._colunas_categoricas)),
            ],
            remainder="drop",
        )

        pipeline_completo = Pipeline(
            steps=[
                ("extrator", ExtratorAtributos()),
                ("transformador", transformador_colunas),
            ]
        )
        return pipeline_completo
