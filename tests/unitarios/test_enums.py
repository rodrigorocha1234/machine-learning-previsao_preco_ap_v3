"""Testes unitários para todos os enums do domínio."""

from imobiliaria_ml.enums_pkg.tipo_modelo import TipoModelo
from imobiliaria_ml.enums_pkg.tipo_ensemble import TipoEnsemble
from imobiliaria_ml.enums_pkg.tipo_escalonador import TipoEscalonador
from imobiliaria_ml.enums_pkg.tipo_evento import TipoEvento
from imobiliaria_ml.enums_pkg.nivel_drift import NivelDrift
from imobiliaria_ml.enums_pkg.origem_peso_voting import OrigemPesoVoting
from imobiliaria_ml.enums_pkg.tipo_carregador import TipoCarregador
from imobiliaria_ml.enums_pkg.tipo_metrica import TipoMetrica
from imobiliaria_ml.enums_pkg.tipo_estimador_final import TipoEstimadorFinal


def test_tipo_modelo_valores() -> None:
    assert TipoModelo.REGRESSAO_LINEAR == "regressao_linear"
    assert TipoModelo.RANDOM_FOREST == "random_forest"
    assert TipoModelo.XGBOOST == "xgboost"
    assert TipoModelo.LIGHTGBM == "lightgbm"
    assert TipoModelo.CATBOOST == "catboost"
    assert len(TipoModelo) == 14


def test_tipo_ensemble_valores() -> None:
    assert TipoEnsemble.VOTING_MEDIA == "voting_media"
    assert TipoEnsemble.VOTING_PONDERADO_RMSE == "voting_ponderado_rmse"
    assert TipoEnsemble.STACKING == "stacking"
    assert TipoEnsemble.BAGGING == "bagging"
    assert len(TipoEnsemble) == 5


def test_tipo_escalonador_valores() -> None:
    assert TipoEscalonador.STANDARD == "standard"
    assert TipoEscalonador.MINMAX == "minmax"
    assert TipoEscalonador.ROBUSTO == "robusto"
    assert TipoEscalonador.SEM_ESCALA == "sem_escala"
    assert len(TipoEscalonador) == 4


def test_tipo_evento_valores() -> None:
    assert TipoEvento.EDA_FINALIZADA == "eda_finalizada"
    assert TipoEvento.MODELO_CAMPEAO == "modelo_campeao"
    assert TipoEvento.FRIEDMAN_FINALIZADO == "friedman_finalizado"
    assert TipoEvento.NEMENYI_FINALIZADO == "nemenyi_finalizado"
    assert len(TipoEvento) == 11


def test_nivel_drift_valores() -> None:
    assert NivelDrift.ESTAVEL == "estavel"
    assert NivelDrift.ATENCAO == "atencao"
    assert NivelDrift.FORTE == "forte"
    assert len(NivelDrift) == 3


def test_origem_peso_voting_valores() -> None:
    assert OrigemPesoVoting.RMSE == "rmse"
    assert OrigemPesoVoting.RANKING == "ranking"


def test_tipo_carregador_valores() -> None:
    assert TipoCarregador.CSV == "csv"
    assert TipoCarregador.EXCEL == "excel"
    assert TipoCarregador.BANCO == "banco"


def test_tipo_metrica_valores() -> None:
    assert TipoMetrica.RMSE == "rmse"
    assert TipoMetrica.MAE == "mae"
    assert TipoMetrica.R2 == "r2"


def test_tipo_estimador_final_valores() -> None:
    assert TipoEstimadorFinal.RIDGE == "ridge"
    assert TipoEstimadorFinal.LASSO == "lasso"
