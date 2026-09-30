from typing import Final

from sklearn.preprocessing import RobustScaler


class NormalizadorNumerico:
    def __init__(self) -> None:
        self._scaler: Final[RobustScaler] = RobustScaler(
            with_centering=True,
            with_scaling=True,
        )

    def obter_scaler(self) -> RobustScaler:
        return self._scaler
