"""Estrutura do resultado das métricas de negócio imobiliário."""

from dataclasses import dataclass, field
import pandas as pd


@dataclass(frozen=True)
class MetricasNegocioResultado:
    """Métricas operacionais e comerciais geradas pelo modelo campeão."""

    cobertura_5: float
    cobertura_10: float
    cobertura_15: float
    mae_reais: float
    vies_medio: float
    desconto_seguro_recomendado: float
    risco_subprecificacao: float
    risco_superprecificacao: float
    margem_negociacao_estimada: float = 0.0
    receita_potencial_perdida: float = 0.0
    tabela_erro_por_faixa: pd.DataFrame = field(default_factory=pd.DataFrame)
    tabela_erro_por_zona: pd.DataFrame = field(default_factory=pd.DataFrame)
    tabela_cobertura_tolerancia: pd.DataFrame = field(default_factory=pd.DataFrame)

