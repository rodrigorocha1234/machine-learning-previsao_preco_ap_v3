"""Exporta o último snapshot do treinamento; não requer bibliotecas externas."""

import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class ExportadorMetricas(BaseHTTPRequestHandler):
    caminho = Path(os.environ.get("METRICAS_TREINO_ARQUIVO", "/metricas/treino.prom"))

    def do_GET(self) -> None:
        rotas = {"/metrics": self.metricas, "/health": self.saude}
        rotas.get(self.path, self.nao_encontrado)()

    def responder(self, codigo: int, conteudo: bytes) -> None:
        self.send_response(codigo)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(conteudo)))
        self.end_headers()
        self.wfile.write(conteudo)

    def metricas(self) -> None:
        try:
            with self.caminho.open("rb") as arquivo:
                conteudo = arquivo.read()
                timestamp = os.fstat(arquivo.fileno()).st_mtime
        except OSError:
            self.responder(503, b"Snapshot de treinamento ainda indisponivel\n")
            return
        marcador = (
            "# HELP apartamentos_snapshot_timestamp Data de gravacao do snapshot de treino\n"
            "# TYPE apartamentos_snapshot_timestamp gauge\n"
            f"apartamentos_snapshot_timestamp {timestamp}\n"
        ).encode()
        self.responder(200, conteudo + marcador)

    def saude(self) -> None:
        self.responder(200, b"ok\n")

    def nao_encontrado(self) -> None:
        self.responder(404, b"Not found\n")

    def log_message(self, format: str, *args: object) -> None:
        pass


def executar_exportador() -> None:
    ThreadingHTTPServer(("0.0.0.0", 8000), ExportadorMetricas).serve_forever()


if __name__ == "__main__":
    executar_exportador()
