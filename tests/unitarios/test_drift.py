"""Testes unitários para detecção e monitoramento de Data Drift."""

import numpy as np
import pandas as pd
from imobiliaria_ml.enums_pkg.nivel_drift import NivelDrift
from imobiliaria_ml.drift_pkg.detector_drift_numerico import DetectorDriftNumerico
from imobiliaria_ml.drift_pkg.detector_drift_categorico import DetectorDriftCategorico
from imobiliaria_ml.drift_pkg.monitor_drift import MonitorDrift


def test_detector_drift_numerico_estavel() -> None:
    np.random.seed(42)
    s1 = np.random.normal(100.0, 10.0, size=500)
    s2 = np.random.normal(100.0, 10.0, size=500)

    detector = DetectorDriftNumerico()
    psi = detector.calcular_psi(s1, s2)
    assert psi < 0.10


def test_monitor_drift_classifica_nivel_drift() -> None:
    np.random.seed(42)
    df_base = pd.DataFrame(
        {
            "Metragem": np.random.normal(100.0, 10.0, size=200),
            "Quartos": np.random.randint(1, 4, size=200),
            "Zona": ["Centro"] * 100 + ["Zona Sul"] * 100,
        }
    )
    # df_atual com desvio forte na Metragem (média 250 vs 100)
    df_atual = pd.DataFrame(
        {
            "Metragem": np.random.normal(250.0, 10.0, size=200),
            "Quartos": np.random.randint(1, 4, size=200),
            "Zona": ["Centro"] * 100 + ["Zona Sul"] * 100,
        }
    )

    monitor = MonitorDrift()
    res = monitor.avaliar(
        df_base=df_base,
        df_atual=df_atual,
        colunas_numericas=["Metragem", "Quartos"],
        colunas_categoricas=["Zona"],
    )

    assert res.drift_detectado
    assert res.nivel_drift in (NivelDrift.ATENCAO, NivelDrift.FORTE)
