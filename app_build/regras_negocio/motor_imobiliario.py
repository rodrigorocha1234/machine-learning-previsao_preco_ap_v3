from typing import Final, override

import pandas as pd

from app_build.regras_negocio.agregador_hierarquico import (
    EstatisticasHierarquicas,
)
from app_build.regras_negocio.avaliador_suficiencia import AvaliadorSuficiencia
from app_build.regras_negocio.contrato_negocio import (
    ContratoNegocio,
    EstatisticasNivel,
    ResultadoImobiliario,
    StatusAmostral,
)
from app_build.regras_negocio.simulador_descontos import SimuladorDescontos


class MotorImobiliario(ContratoNegocio):
    def __init__(
        self,
        estatisticas: EstatisticasHierarquicas,
        avaliador_suficiencia: AvaliadorSuficiencia | None = None,
        simulador_descontos: SimuladorDescontos | None = None,
    ) -> None:
        self._estatisticas: Final[EstatisticasHierarquicas] = estatisticas
        self._avaliador: Final[AvaliadorSuficiencia] = (
            avaliador_suficiencia or AvaliadorSuficiencia()
        )
        self._simulador: Final[SimuladorDescontos] = (
            simulador_descontos or SimuladorDescontos()
        )

    @property
    def estatisticas(self) -> EstatisticasHierarquicas:
        return self._estatisticas

    @override
    def enriquecer_previsao(
        self,
        valor_previsto: float,
        metragem: float,
        zona: str,
        bairro: str,
    ) -> ResultadoImobiliario:
        metragem_segura = max(metragem, 1.0)
        valor_m2_previsto = float(valor_previsto / metragem_segura)

        estat_global = self._estatisticas.estatisticas_global
        estat_zona = self._estatisticas.tabela_zonas.get(
            zona,
            EstatisticasNivel(
                nivel="ZONA",
                nome_chave=zona,
                media_valor=estat_global.media_valor,
                mediana_valor=estat_global.mediana_valor,
                media_valor_m2=estat_global.media_valor_m2,
                mediana_valor_m2=estat_global.mediana_valor_m2,
                desvio_valor_m2=estat_global.desvio_valor_m2,
                total_amostras=0,
                status_amostral=StatusAmostral.NAO_DISPONIVEL,
            ),
        )
        estat_bairro = self._estatisticas.tabela_bairros.get(
            bairro,
            EstatisticasNivel(
                nivel="BAIRRO",
                nome_chave=bairro,
                media_valor=estat_zona.media_valor,
                mediana_valor=estat_zona.mediana_valor,
                media_valor_m2=estat_zona.media_valor_m2,
                mediana_valor_m2=estat_zona.mediana_valor_m2,
                desvio_valor_m2=estat_zona.desvio_valor_m2,
                total_amostras=0,
                status_amostral=StatusAmostral.NAO_DISPONIVEL,
            ),
        )

        decisao = self._avaliador.resolver_ancora(
            estat_bairro, estat_zona, estat_global
        )

        # Índices e desvios estritamente nos 3 níveis: GLOBAL -> ZONA -> BAIRRO
        indice_imovel_global = float(
            valor_m2_previsto / max(estat_global.mediana_valor_m2, 1.0)
        )
        indice_imovel_zona = float(
            valor_m2_previsto / max(estat_zona.mediana_valor_m2, 1.0)
        )
        indice_imovel_bairro = float(
            valor_m2_previsto / max(estat_bairro.mediana_valor_m2, 1.0)
        )

        indice_zona_global = float(
            estat_zona.mediana_valor_m2 / max(estat_global.mediana_valor_m2, 1.0)
        )
        indice_bairro_zona = float(
            estat_bairro.mediana_valor_m2 / max(estat_zona.mediana_valor_m2, 1.0)
        )
        indice_bairro_global = float(
            estat_bairro.mediana_valor_m2 / max(estat_global.mediana_valor_m2, 1.0)
        )

        dif_perc_global = float(
            (
                (valor_previsto - estat_global.media_valor)
                / max(estat_global.media_valor, 1.0)
            )
            * 100.0
        )
        dif_perc_zona = float(
            (
                (valor_previsto - estat_zona.media_valor)
                / max(estat_zona.media_valor, 1.0)
            )
            * 100.0
        )
        dif_perc_bairro = float(
            (
                (valor_previsto - estat_bairro.media_valor)
                / max(estat_bairro.media_valor, 1.0)
            )
            * 100.0
        )

        simulacao = self._simulador.simular(valor_previsto)

        return ResultadoImobiliario(
            valor_previsto=valor_previsto,
            valor_m2_previsto=valor_m2_previsto,
            estatisticas_global=estat_global,
            estatisticas_zona=estat_zona,
            estatisticas_bairro=estat_bairro,
            indice_imovel_global=indice_imovel_global,
            indice_imovel_zona=indice_imovel_zona,
            indice_imovel_bairro=indice_imovel_bairro,
            indice_zona_global=indice_zona_global,
            indice_bairro_zona=indice_bairro_zona,
            indice_bairro_global=indice_bairro_global,
            diferenca_percentual_global=dif_perc_global,
            diferenca_percentual_zona=dif_perc_zona,
            diferenca_percentual_bairro=dif_perc_bairro,
            nivel_referencia_utilizado=decisao.nivel_escolhido,
            status_final_amostra=decisao.status_resolvido,
            desconto_moderado_5=simulacao.desconto_moderado_5,
            desconto_agressivo_10=simulacao.desconto_agressivo_10,
            desconto_queima_15=simulacao.desconto_queima_15,
            faixa_segura_piso=simulacao.faixa_segura_piso,
            faixa_segura_teto=simulacao.faixa_segura_teto,
        )

    @override
    def enriquecer_dataframe(
        self,
        dados_x: pd.DataFrame,
        vetor_previsoes: pd.Series,
    ) -> pd.DataFrame:
        previsoes_m2 = vetor_previsoes / dados_x["Metragem"].clip(lower=1.0)

        # Mapeamento vetorizado para Global, Zona e Bairro
        med_m2_glob = self._estatisticas.estatisticas_global.mediana_valor_m2
        media_val_glob = self._estatisticas.estatisticas_global.media_valor

        mapa_zona_mediana_m2 = {
            z: estat.mediana_valor_m2
            for z, estat in self._estatisticas.tabela_zonas.items()
        }
        mapa_zona_media_val = {
            z: estat.media_valor for z, estat in self._estatisticas.tabela_zonas.items()
        }

        mapa_bairro_mediana_m2 = {
            b: estat.mediana_valor_m2
            for b, estat in self._estatisticas.tabela_bairros.items()
        }
        mapa_bairro_media_val = {
            b: estat.media_valor
            for b, estat in self._estatisticas.tabela_bairros.items()
        }

        zona_med_m2_series = (
            dados_x["Zona"].map(mapa_zona_mediana_m2).fillna(med_m2_glob)
        )
        zona_med_val_series = (
            dados_x["Zona"].map(mapa_zona_media_val).fillna(media_val_glob)
        )

        bairro_med_m2_series = (
            dados_x["Bairro"].map(mapa_bairro_mediana_m2).fillna(zona_med_m2_series)
        )
        bairro_med_val_series = (
            dados_x["Bairro"].map(mapa_bairro_media_val).fillna(zona_med_val_series)
        )

        # Mediana de valor total por nível (usada como âncora para descontos/faixas)
        med_val_glob = self._estatisticas.estatisticas_global.mediana_valor

        mapa_zona_mediana_val = {
            z: estat.mediana_valor
            for z, estat in self._estatisticas.tabela_zonas.items()
        }
        mapa_bairro_mediana_val = {
            b: estat.mediana_valor
            for b, estat in self._estatisticas.tabela_bairros.items()
        }

        zona_med_val_total = (
            dados_x["Zona"].map(mapa_zona_mediana_val).fillna(med_val_glob)
        )
        bairro_med_val_total = (
            dados_x["Bairro"]
            .map(mapa_bairro_mediana_val)
            .fillna(zona_med_val_total)
        )

        resultado = dados_x.assign(
            valor_previsto=vetor_previsoes,
            valor_m2_previsto=previsoes_m2,
            # --- Nível Global ---
            global_mediana_mercado=med_val_glob,
            global_media_mercado=media_val_glob,
            global_mediana_m2_mercado=med_m2_glob,
            indice_imovel_global=previsoes_m2 / med_m2_glob,
            diferenca_perc_global=(
                (vetor_previsoes - media_val_glob) / media_val_glob
            ) * 100.0,
            global_desconto_5=med_val_glob * 0.95,
            global_desconto_10=med_val_glob * 0.90,
            global_desconto_15=med_val_glob * 0.85,
            global_faixa_segura_piso=med_val_glob * 0.90,
            global_faixa_segura_teto=med_val_glob,
            # --- Nível Zona ---
            zona_mediana_mercado=zona_med_val_total,
            zona_media_mercado=zona_med_val_series,
            zona_mediana_m2_mercado=zona_med_m2_series,
            indice_imovel_zona=previsoes_m2 / zona_med_m2_series,
            diferenca_perc_zona=(
                (vetor_previsoes - zona_med_val_series) / zona_med_val_series
            ) * 100.0,
            zona_desconto_5=zona_med_val_total * 0.95,
            zona_desconto_10=zona_med_val_total * 0.90,
            zona_desconto_15=zona_med_val_total * 0.85,
            zona_faixa_segura_piso=zona_med_val_total * 0.90,
            zona_faixa_segura_teto=zona_med_val_total,
            # --- Nível Bairro ---
            bairro_mediana_mercado=bairro_med_val_total,
            bairro_media_mercado=bairro_med_val_series,
            bairro_mediana_m2_mercado=bairro_med_m2_series,
            indice_imovel_bairro=previsoes_m2 / bairro_med_m2_series,
            diferenca_perc_bairro=(
                (vetor_previsoes - bairro_med_val_series) / bairro_med_val_series
            ) * 100.0,
            bairro_desconto_5=bairro_med_val_total * 0.95,
            bairro_desconto_10=bairro_med_val_total * 0.90,
            bairro_desconto_15=bairro_med_val_total * 0.85,
            bairro_faixa_segura_piso=bairro_med_val_total * 0.90,
            bairro_faixa_segura_teto=bairro_med_val_total,
        )
        return resultado
