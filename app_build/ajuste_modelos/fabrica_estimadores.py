from collections.abc import Callable, Mapping
from typing import Final

import lightgbm as lgb
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

CriadorEstimador = Callable[[Mapping[str, object]], BaseEstimator]

REGISTRO_ESTIMADORES: Final[Mapping[str, CriadorEstimador]] = {
    "linear_regression": lambda params: LinearRegression(**params),
    "ridge": lambda params: Ridge(**params),
    "lasso": lambda params: Lasso(**params),
    "elastic_net": lambda params: ElasticNet(**params),
    "arvore_decisao": lambda params: DecisionTreeRegressor(**params),
    "random_forest": lambda params: RandomForestRegressor(**params),
    "svr": lambda params: SVR(**params),
    "rede_neural": lambda params: MLPRegressor(**params),
    "xgboost": lambda params: xgb.XGBRegressor(**params),
    "lightgbm": lambda params: lgb.LGBMRegressor(verbose=-1, **dict(params)),  # type: ignore[arg-type]
}


class FabricaEstimadores:
    @staticmethod
    def criar_estimador(
        nome_modelo: str, parametros: Mapping[str, object] | None = None
    ) -> BaseEstimator:
        params_dict = dict(parametros or {})
        try:
            construtor = REGISTRO_ESTIMADORES[nome_modelo]
            return construtor(params_dict)
        except KeyError as erro:
            raise ValueError(
                f"Modelo nao reconhecido no catalogo: '{nome_modelo}'"
            ) from erro
