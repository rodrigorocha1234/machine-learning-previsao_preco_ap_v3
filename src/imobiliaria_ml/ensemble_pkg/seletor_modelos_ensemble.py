"""Filtro de modelos para ensemble restrito ao grupo estatisticamente elegível."""

from sklearn.pipeline import Pipeline
from ..validacao_pkg.grupo_elegivel import GrupoElegivel


class SeletorModelosEnsemble:
    """Filtra e prepara estimadores elegíveis para composição de ensembles."""

    def filtrar_elegiveis(
        self,
        todos_modelos: dict[str, Pipeline],
        grupo: GrupoElegivel,
    ) -> list[tuple[str, Pipeline]]:
        """Retorna tuplas (nome, pipeline) apenas dos modelos pertencentes ao grupo elegível.

        Parameters
        ----------
        todos_modelos : dict[str, Pipeline]
            Coleção completa de pipelines candidatas.
        grupo : GrupoElegivel
            Grupo estatisticamente indistinguível do melhor modelo.

        Returns
        -------
        list[tuple[str, Pipeline]]
            Lista de tuplas dos modelos elegíveis prontos para o ensemble.
        """
        return [
            (nome, pipeline)
            for nome, pipeline in todos_modelos.items()
            if nome in grupo.modelos_elegiveis
        ]
