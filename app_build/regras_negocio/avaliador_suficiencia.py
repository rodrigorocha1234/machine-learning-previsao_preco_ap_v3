from dataclasses import dataclass

from app_build.regras_negocio.contrato_negocio import EstatisticasNivel, StatusAmostral


@dataclass(frozen=True)
class DecisaoReferencia:
    nivel_escolhido: str
    estatistica_ancora: EstatisticasNivel
    status_resolvido: StatusAmostral


class AvaliadorSuficiencia:
    def resolver_ancora(
        self,
        estat_bairro: EstatisticasNivel,
        estat_zona: EstatisticasNivel,
        estat_global: EstatisticasNivel,
    ) -> DecisaoReferencia:
        tem_bairro_suficiente = (
            estat_bairro.status_amostral == StatusAmostral.SUFICIENTE
        )
        tem_zona_suficiente = estat_zona.status_amostral == StatusAmostral.SUFICIENTE

        # Mapeamento declarativo da cadeia de fallback: (bairro_ok, zona_ok) -> Decisão
        tabela_fallback: dict[tuple[bool, bool], DecisaoReferencia] = {
            (True, True): DecisaoReferencia(
                nivel_escolhido="BAIRRO",
                estatistica_ancora=estat_bairro,
                status_resolvido=StatusAmostral.SUFICIENTE,
            ),
            (True, False): DecisaoReferencia(
                nivel_escolhido="BAIRRO",
                estatistica_ancora=estat_bairro,
                status_resolvido=StatusAmostral.SUFICIENTE,
            ),
            (False, True): DecisaoReferencia(
                nivel_escolhido="ZONA",
                estatistica_ancora=estat_zona,
                status_resolvido=StatusAmostral.AMOSTRA_INSUFICIENTE,
            ),
            (False, False): DecisaoReferencia(
                nivel_escolhido="GLOBAL",
                estatistica_ancora=estat_global,
                status_resolvido=StatusAmostral.AMOSTRA_INSUFICIENTE,
            ),
        }

        return tabela_fallback[(tem_bairro_suficiente, tem_zona_suficiente)]
