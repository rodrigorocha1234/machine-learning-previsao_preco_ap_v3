import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class EventoLog:
    nivel: str
    mensagem: str
    servico: str


class EmissorLogs:
    def __init__(self, url_loki: str = "http://localhost:3100/loki/api/v1/push") -> None:
        self._url_loki = url_loki

    def emitir_evento(self, evento: EventoLog) -> bool:
        timestamp_ns = str(time.time_ns())
        payload = {
            "streams": [
                {
                    "stream": {
                        "container": evento.servico,
                        "level": evento.nivel.lower(),
                        "service_name": evento.servico,
                    },
                    "values": [
                        [
                            timestamp_ns,
                            f"{time.strftime('%Y-%m-%d %H:%M:%S')} {evento.nivel.upper()}: [{evento.servico}] {evento.mensagem}",
                        ]
                    ],
                }
            ]
        }
        dados = json.dumps(payload).encode("utf-8")
        requisicao = urllib.request.Request(
            self._url_loki,
            data=dados,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(requisicao, timeout=3.0) as resposta:
                return resposta.status in (200, 204)
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def emitir_lote(self, eventos: list[EventoLog]) -> int:
        timestamp_base = time.time_ns()
        streams_agrupados: dict[tuple[str, str], list[list[str]]] = {}

        for indice, ev in enumerate(eventos):
            chave = (ev.servico, ev.nivel.lower())
            ts = str(timestamp_base + (indice * 1000000))
            linha = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {ev.nivel.upper()}: [{ev.servico}] {ev.mensagem}"
            streams_agrupados.setdefault(chave, []).append([ts, linha])

        payload = {
            "streams": [
                {
                    "stream": {
                        "container": serv,
                        "level": niv,
                        "service_name": serv,
                    },
                    "values": vals,
                }
                for (serv, niv), vals in streams_agrupados.items()
            ]
        }
        dados = json.dumps(payload).encode("utf-8")
        requisicao = urllib.request.Request(
            self._url_loki,
            data=dados,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(requisicao, timeout=5.0) as resposta:
                status_ok = resposta.status in (200, 204)
                fator = {True: len(eventos), False: 0}
                return fator[status_ok]
        except (urllib.error.URLError, TimeoutError, OSError):
            return 0
