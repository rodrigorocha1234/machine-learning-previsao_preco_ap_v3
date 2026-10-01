import numpy as np
import pandas as pd
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info

from app_build.observabilidade_metricas.distribuicao_serving import DistribuicaoServing


class MetricasServing:
    def __init__(self, zonas: set[str], bairros: set[str]) -> None:
        self.registry = CollectorRegistry()
        self.zonas = zonas
        self.bairros = bairros
        self.requisicoes = Counter(
            "apartamentos_serving_requisicoes",
            "Requisicoes HTTP de inferencia",
            ["status"],
            registry=self.registry,
        )
        for status in ("2xx", "3xx", "4xx", "5xx"):
            self.requisicoes.labels(status)
        self.latencia = Histogram(
            "apartamentos_serving_latencia_segundos",
            "Tempo HTTP completo de inferencia",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
            registry=self.registry,
        )
        self.lotes = Histogram(
            "apartamentos_serving_lote_imoveis",
            "Quantidade de imoveis em requisicoes JSON reconhecidas",
            buckets=(1, 2, 5, 10, 25, 50, 100, 500, 1000),
            registry=self.registry,
        )
        self.entradas = Counter(
            "apartamentos_serving_entradas",
            "Imoveis recebidos em JSON reconhecido",
            registry=self.registry,
        )
        self.problemas = Counter(
            "apartamentos_serving_entrada_problemas",
            "Ocorrencias por campo; um imovel pode ter varios problemas",
            ["campo", "motivo"],
            registry=self.registry,
        )
        self.referencias = Counter(
            "apartamentos_serving_referencia",
            "Nivel de referencia disponivel usado pelo enriquecimento vetorizado",
            ["nivel"],
            registry=self.registry,
        )
        self.falhas = Counter(
            "apartamentos_serving_telemetria_falhas",
            "Falhas de interpretacao da telemetria sem alterar a resposta",
            registry=self.registry,
        )
        self.modelo = Info(
            "apartamentos_serving_modelo",
            "Versao efetivamente carregada pelo serving",
            registry=self.registry,
        )
        self.carregado = Gauge(
            "apartamentos_serving_carregado_timestamp",
            "Instante de carga do modelo",
            registry=self.registry,
        )
        self.registrado = Gauge(
            "apartamentos_serving_registrado_timestamp",
            "Criacao da versao no Registry",
            registry=self.registry,
        )
        self.treinado = Gauge(
            "apartamentos_serving_treinado_timestamp",
            "Fim do run de treino associado",
            registry=self.registry,
        )
        self.distribuicao = DistribuicaoServing()
        self.registry.register(self.distribuicao)
        for nivel in ("bairro", "zona", "global"):
            self.referencias.labels(nivel)
        for campo in (
            "Zona",
            "Bairro",
            "Metragem",
            "Quartos",
            "Banheiros",
            "Vagas_Garagem",
        ):
            self.problemas.labels(campo, "ausente")
        for campo in ("Zona", "Bairro"):
            self.problemas.labels(campo, "desconhecida")
        self.problemas.labels("Metragem", "invalida")

    def registrar_entrada(self, dados: pd.DataFrame) -> None:
        self.entradas.inc(len(dados))
        self.lotes.observe(len(dados))
        campos = ["Zona", "Bairro", "Metragem", "Quartos", "Banheiros", "Vagas_Garagem"]
        base = dados.reindex(columns=campos)
        for campo, total in base.isna().sum().items():
            self.problemas.labels(str(campo), "ausente").inc(int(total))
        for campo, conhecidos in (("Zona", self.zonas), ("Bairro", self.bairros)):
            total = (base[campo].notna() & ~base[campo].isin(conhecidos)).sum()
            self.problemas.labels(campo, "desconhecida").inc(int(total))
        area = pd.to_numeric(base["Metragem"], errors="coerce")
        invalidas = base["Metragem"].notna() & (~np.isfinite(area) | (area <= 0))
        self.problemas.labels("Metragem", "invalida").inc(int(invalidas.sum()))

    def registrar_saida(self, dados: pd.DataFrame) -> None:
        zona_conhecida = dados["Zona"].isin(self.zonas)
        bairro_conhecido = dados["Bairro"].isin(self.bairros)
        base = pd.DataFrame(
            {
                "zona": dados["Zona"].where(zona_conhecida, "DESCONHECIDA"),
                "bairro": dados["Bairro"].where(bairro_conhecido, "DESCONHECIDO"),
                "valor": dados["valor_previsto"],
                "valor_m2": dados["valor_m2_previsto"],
            }
        )
        self.distribuicao.registrar(base)
        niveis = pd.Series(
            np.select(
                [bairro_conhecido, zona_conhecida], ["bairro", "zona"], default="global"
            )
        )
        for nivel, total in niveis.value_counts().items():
            self.referencias.labels(str(nivel)).inc(int(total))
