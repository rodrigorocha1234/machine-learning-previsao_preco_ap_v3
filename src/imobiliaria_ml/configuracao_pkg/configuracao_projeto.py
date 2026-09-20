"""Dataclass estruturada de configuração do projeto convertida em Enums."""

from dataclasses import dataclass, field
from ..enums_pkg.tipo_metrica import TipoMetrica
from ..enums_pkg.tipo_ensemble import TipoEnsemble
from ..enums_pkg.tipo_estimador_final import TipoEstimadorFinal


@dataclass(frozen=True)
class ConfiguracaoProjeto:
    """Configuração tipada de ponta a ponta sem tipos soltos ou Any."""

    # Projeto
    alvo: str = "Valor_da_Venda"
    seed: int = 42

    # Validação
    n_splits: int = 5
    n_repeats: int = 30
    alpha_friedman: float = 0.05
    alpha_nemenyi: float = 0.05
    metrica_primaria: TipoMetrica = TipoMetrica.RMSE

    # Ensemble
    usar_votacao: bool = False
    tecnicas_ensemble: tuple[TipoEnsemble, ...] = field(
        default_factory=lambda: (
            TipoEnsemble.VOTING_MEDIA,
            TipoEnsemble.VOTING_PONDERADO_RMSE,
            TipoEnsemble.VOTING_PONDERADO_RANKING,
            TipoEnsemble.STACKING,
            TipoEnsemble.BAGGING,
        )
    )
    stacking_estimador_final: TipoEstimadorFinal = TipoEstimadorFinal.RIDGE
    bagging_n_estimators: int = 100
    bagging_max_samples: float = 0.8
    bagging_bootstrap: bool = True

    # MLflow
    mlflow_tracking_uri_env: str = "MLFLOW_TRACKING_URI"
    mlflow_experimento: str = "previsao-preco-imoveis"
    mlflow_nome_modelo: str = "preco-imoveis"
    mlflow_persistencia_local_artefatos: bool = False

    # Negócio
    desconto_minimo: float = 0.0
    desconto_maximo_automatico: float = 5.0
    desconto_maximo_com_aprovacao: float = 10.0

    # Drift
    drift_psi_atencao: float = 0.10
    drift_psi_forte: float = 0.25
