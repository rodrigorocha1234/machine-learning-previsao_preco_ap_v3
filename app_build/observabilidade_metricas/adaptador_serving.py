import json
import time

import pandas as pd
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app_build.observabilidade_metricas.metricas_serving import MetricasServing


class AdaptadorServing:
    """Adaptador ASGI de telemetria; o MLflow mantém validação e inferência."""

    def __init__(self, app: ASGIApp, metricas: MetricasServing) -> None:
        self.app = app
        self.metricas = metricas

    def _registrar_json(self, corpo: bytes, saida: bool) -> None:
        try:
            payload = json.loads(corpo)
            if saida:
                self.metricas.registrar_saida(pd.DataFrame(payload["predictions"]))
            elif "dataframe_records" in payload:
                self.metricas.registrar_entrada(
                    pd.DataFrame(payload["dataframe_records"])
                )
            elif "dataframe_split" in payload:
                self.metricas.registrar_entrada(
                    pd.DataFrame(**payload["dataframe_split"])
                )
        except (ValueError, TypeError, KeyError, OverflowError):
            self.metricas.falhas.inc()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"] == "/metrics":
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", CONTENT_TYPE_LATEST.encode())],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": generate_latest(self.metricas.registry),
                }
            )
            return
        if scope["type"] != "http" or scope["path"] != "/invocations":
            await self.app(scope, receive, send)
            return
        inicio = time.perf_counter()
        entrada = bytearray()
        saida = bytearray()
        status = 500

        async def receber() -> Message:
            mensagem = await receive()
            if mensagem["type"] == "http.request":
                entrada.extend(mensagem.get("body", b""))
            return mensagem

        async def enviar(mensagem: Message) -> None:
            nonlocal status
            if mensagem["type"] == "http.response.start":
                status = mensagem["status"]
            elif mensagem["type"] == "http.response.body":
                saida.extend(mensagem.get("body", b""))
            await send(mensagem)

        try:
            await self.app(scope, receber, enviar)
        finally:
            self.metricas.latencia.observe(time.perf_counter() - inicio)
            self.metricas.requisicoes.labels(f"{status // 100}xx").inc()
            cabecalhos = dict(scope.get("headers", []))
            if entrada and b"application/json" in cabecalhos.get(b"content-type", b""):
                self._registrar_json(bytes(entrada), False)
            if 200 <= status < 300 and saida:
                self._registrar_json(bytes(saida), True)
