"""Tratador de variáveis categóricas com imputação e One-Hot Encoding."""

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class TratadorCategorico:
    """Monta a pipeline para tratamento de variáveis categóricas."""

    def criar_pipeline(self) -> Pipeline:
        """Cria o pipeline com SimpleImputer(strategy='most_frequent') e OneHotEncoder.

        Returns
        -------
        Pipeline
            Pipeline do sklearn para colunas categóricas.
        """
        return Pipeline(
            steps=[
                ("imputador", SimpleImputer(strategy="most_frequent")),
                ("codificador", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
