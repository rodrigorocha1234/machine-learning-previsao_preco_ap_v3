from collections.abc import Iterator
from threading import Lock

import numpy as np
import pandas as pd
from prometheus_client.core import HistogramMetricFamily
from prometheus_client.utils import floatToGoString


class DistribuicaoServing:
    """Histogramas cumulativos por localidade, atualizados por agregações vetorizadas."""

    def __init__(self) -> None:
        self._limites = {
            "valor": np.array(
                [
                    100000,
                    200000,
                    300000,
                    500000,
                    750000,
                    1000000,
                    2000000,
                    5000000,
                    np.inf,
                ]
            ),
            "valor_m2": np.array(
                [1000, 2000, 4000, 6000, 8000, 10000, 15000, 25000, np.inf]
            ),
        }
        self._dados: dict[tuple[str, str, str], tuple[np.ndarray, float]] = {}
        self._lock = Lock()

    def registrar(self, dados: pd.DataFrame) -> None:
        for campo, limites in self._limites.items():
            valores = pd.to_numeric(dados[campo], errors="coerce")
            validos = valores.notna() & np.isfinite(valores) & (valores >= 0)
            base = dados.loc[validos, ["zona", "bairro"]].copy()
            base["soma"] = valores[validos]
            colunas = [f"b{i}" for i in range(len(limites))]
            base[colunas] = (valores[validos].to_numpy()[:, None] <= limites).astype(
                int
            )
            grupos = base.groupby(["zona", "bairro"])[[*colunas, "soma"]].sum()
            with self._lock:
                for (zona, bairro), agregado in grupos.to_dict(orient="index").items():
                    chave = (campo, str(zona), str(bairro))
                    contagens, soma = self._dados.get(
                        chave, (np.zeros(len(limites)), 0.0)
                    )
                    self._dados[chave] = (
                        contagens + np.array([agregado[c] for c in colunas]),
                        soma + float(agregado["soma"]),
                    )

    def collect(self) -> Iterator[HistogramMetricFamily]:
        with self._lock:
            dados = dict(self._dados)
        for campo, limites in self._limites.items():
            familia = HistogramMetricFamily(
                f"apartamentos_serving_{campo}",
                "Distribuicao das previsoes reais por imovel atendido",
                labels=["zona", "bairro"],
            )
            grupos = filter(lambda item: item[0][0] == campo, dados.items())
            for (_, zona, bairro), (contagens, soma) in grupos:
                familia.add_metric(
                    [zona, bairro],
                    [
                        (floatToGoString(limite), float(contagem))
                        for limite, contagem in zip(limites, contagens)
                    ],
                    sum_value=soma,
                )
            yield familia
