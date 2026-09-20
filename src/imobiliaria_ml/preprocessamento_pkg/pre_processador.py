"""Montador do pipeline de pré-processamento com ColumnTransformer."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from .estrategia_escalonamento import EstrategiaEscalonamento
from .tratador_categorico import TratadorCategorico


class PreProcessador:
    """Responsável por construir o ColumnTransformer para variáveis numéricas e categóricas."""

    def __init__(
        self,
        estrategia_escalonamento: EstrategiaEscalonamento,
        tratador_categorico: TratadorCategorico | None = None,
    ) -> None:
        """Inicializa com a estratégia de escalonamento e tratador categórico.

        Parameters
        ----------
        estrategia_escalonamento : EstrategiaEscalonamento
            Estratégia para atributos numéricos.
        tratador_categorico : TratadorCategorico | None
            Tratador para atributos categóricos. Se None, instancia o padrão.
        """
        self._estrategia_escalonamento = estrategia_escalonamento
        self._tratador_categorico = tratador_categorico or TratadorCategorico()

    def construir_pipeline(
        self,
        colunas_numericas: list[str],
        colunas_categoricas: list[str],
    ) -> ColumnTransformer:
        """Monta o ColumnTransformer completo unindo numéricas e categóricas.

        Parameters
        ----------
        colunas_numericas : list[str]
            Lista de nomes das colunas numéricas.
        colunas_categoricas : list[str]
            Lista de nomes das colunas categóricas.

        Returns
        -------
        ColumnTransformer
            Transformador sklearn pronto para ajuste e transformação.
        """
        pipeline_numerica = Pipeline(
            steps=[
                ("imputador", SimpleImputer(strategy="median")),
                ("escalonador", self._estrategia_escalonamento.criar_transformador()),
            ]
        )

        pipeline_categorica = self._tratador_categorico.criar_pipeline()

        transformadores = []
        if colunas_numericas:
            transformadores.append(("num", pipeline_numerica, colunas_numericas))
        if colunas_categoricas:
            transformadores.append(("cat", pipeline_categorica, colunas_categoricas))

        return ColumnTransformer(
            transformers=transformadores,
            remainder="drop",
        )
