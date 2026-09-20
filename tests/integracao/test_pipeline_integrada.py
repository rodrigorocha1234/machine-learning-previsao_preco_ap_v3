"""Teste de integração ponta a ponta do PipelineTreinamento."""

from pathlib import Path
import pandas as pd
from imobiliaria_ml.configuracao_pkg.configuracao_projeto import ConfiguracaoProjeto
from imobiliaria_ml.enums_pkg.tipo_modelo import TipoModelo
from imobiliaria_ml.pipeline_pkg.pipeline_treinamento import PipelineTreinamento


def test_pipeline_execucao_completa(
    df_imoveis_sintetico: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Executa o ciclo oficial completo de treinamento, validação cruzada, Friedman e holdout."""
    csv_teste = tmp_path / "imoveis_teste.csv"
    df_imoveis_sintetico.to_csv(csv_teste, index=False)

    # Configuração compacta para agilidade de teste
    config = ConfiguracaoProjeto(
        n_splits=3,
        n_repeats=2,  # 2 repetições rápidas para validação em teste
        usar_votacao=True,
    )

    # Executa com três modelos para exercitar Friedman e Nemenyi
    modelos = [TipoModelo.REGRESSAO_LINEAR, TipoModelo.RIDGE, TipoModelo.LASSO]
    pipeline = PipelineTreinamento(
        configuracao=config,
        modelos_para_executar=modelos,
    )

    resultado = pipeline.executar(csv_teste)

    assert resultado.modelo is not None
    assert resultado.metricas.rmse > 0
    assert resultado.metricas_negocio is not None
    assert resultado.metricas_negocio.cobertura_15 >= 0.0
    assert resultado.explicabilidade is not None
