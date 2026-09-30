from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from app_build.regras_negocio.contrato_negocio import EstatisticasNivel, StatusAmostral


@dataclass(frozen=True)
class EstatisticasHierarquicas:
    estatisticas_global: EstatisticasNivel
    tabela_zonas: dict[str, EstatisticasNivel]
    tabela_bairros: dict[str, EstatisticasNivel]


class AgregadorHierarquico:
    def __init__(self, minima_zona: int = 30, minima_bairro: int = 20) -> None:
        self._min_zona: Final[int] = minima_zona
        self._min_bairro: Final[int] = minima_bairro

    def calcular_estatisticas(
        self, dados_treino: pd.DataFrame, target: str = "Valor_da_Venda"
    ) -> EstatisticasHierarquicas:
        dados_com_m2 = dados_treino.assign(
            valor_m2_calculado=dados_treino[target]
            / dados_treino["Metragem"].clip(lower=1.0)
        )

        # Nível 1: GLOBAL (Ribeirão Preto)
        media_val_glob = float(dados_com_m2[target].mean())
        mediana_val_glob = float(dados_com_m2[target].median())
        media_m2_glob = float(dados_com_m2["valor_m2_calculado"].mean())
        mediana_m2_glob = float(dados_com_m2["valor_m2_calculado"].median())
        desvio_m2_glob = float(dados_com_m2["valor_m2_calculado"].std())
        total_glob = len(dados_com_m2)

        estatisticas_global = EstatisticasNivel(
            nivel="GLOBAL",
            nome_chave="Ribeirão Preto",
            media_valor=media_val_glob,
            mediana_valor=mediana_val_glob,
            media_valor_m2=media_m2_glob,
            mediana_valor_m2=mediana_m2_glob,
            desvio_valor_m2=desvio_m2_glob,
            total_amostras=total_glob,
            status_amostral=StatusAmostral.SUFICIENTE,
        )

        # Nível 2: ZONA
        agg_zona = dados_com_m2.groupby("Zona").agg(
            media_val=(target, "mean"),
            mediana_val=(target, "median"),
            media_m2=("valor_m2_calculado", "mean"),
            mediana_m2=("valor_m2_calculado", "median"),
            desvio_m2=("valor_m2_calculado", "std"),
            contagem=(target, "count"),
        )

        dicionario_zonas = agg_zona.to_dict(orient="index")
        tabela_zonas: dict[str, EstatisticasNivel] = {}
        for zona_nome, linha in dicionario_zonas.items():
            qtd = int(linha["contagem"])
            status = (
                StatusAmostral.SUFICIENTE
                if qtd >= self._min_zona
                else StatusAmostral.AMOSTRA_INSUFICIENTE
            )
            desvio = (
                float(linha["desvio_m2"]) if not np.isnan(linha["desvio_m2"]) else 0.0
            )
            tabela_zonas[str(zona_nome)] = EstatisticasNivel(
                nivel="ZONA",
                nome_chave=str(zona_nome),
                media_valor=float(linha["media_val"]),
                mediana_valor=float(linha["mediana_val"]),
                media_valor_m2=float(linha["media_m2"]),
                mediana_valor_m2=float(linha["mediana_m2"]),
                desvio_valor_m2=desvio,
                total_amostras=qtd,
                status_amostral=status,
            )

        # Nível 3: BAIRRO
        agg_bairro = dados_com_m2.groupby("Bairro").agg(
            media_val=(target, "mean"),
            mediana_val=(target, "median"),
            media_m2=("valor_m2_calculado", "mean"),
            mediana_m2=("valor_m2_calculado", "median"),
            desvio_m2=("valor_m2_calculado", "std"),
            contagem=(target, "count"),
        )

        dicionario_bairros = agg_bairro.to_dict(orient="index")
        tabela_bairros: dict[str, EstatisticasNivel] = {}
        for bairro_nome, linha in dicionario_bairros.items():
            qtd = int(linha["contagem"])
            status = (
                StatusAmostral.SUFICIENTE
                if qtd >= self._min_bairro
                else StatusAmostral.AMOSTRA_INSUFICIENTE
            )
            desvio = (
                float(linha["desvio_m2"]) if not np.isnan(linha["desvio_m2"]) else 0.0
            )
            tabela_bairros[str(bairro_nome)] = EstatisticasNivel(
                nivel="BAIRRO",
                nome_chave=str(bairro_nome),
                media_valor=float(linha["media_val"]),
                mediana_valor=float(linha["mediana_val"]),
                media_valor_m2=float(linha["media_m2"]),
                mediana_valor_m2=float(linha["mediana_m2"]),
                desvio_valor_m2=desvio,
                total_amostras=qtd,
                status_amostral=status,
            )

        return EstatisticasHierarquicas(
            estatisticas_global=estatisticas_global,
            tabela_zonas=tabela_zonas,
            tabela_bairros=tabela_bairros,
        )
