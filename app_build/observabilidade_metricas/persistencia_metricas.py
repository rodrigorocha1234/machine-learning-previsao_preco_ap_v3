import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from prometheus_client import REGISTRY, CollectorRegistry, generate_latest


class PersistenciaMetricas:
    """Publica um snapshot atômico, preservando o anterior se a coleta falhar."""

    def __init__(
        self, caminho: Path | None = None, registry: CollectorRegistry = REGISTRY
    ) -> None:
        padrao = (
            Path(__file__).resolve().parents[2] / "observabilidade_data" / "treino.prom"
        )
        self.caminho = caminho or Path(
            os.environ.get("METRICAS_TREINO_ARQUIVO", str(padrao))
        )
        self.registry = registry

    def salvar(self) -> None:
        conteudo = generate_latest(self.registry)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            dir=self.caminho.parent, prefix=".treino-", delete=False
        ) as temporario:
            nome = Path(temporario.name)
            try:
                temporario.write(conteudo)
                temporario.flush()
                os.fsync(temporario.fileno())
                temporario.close()
                nome.chmod(0o644)
                nome.replace(self.caminho)
            finally:
                nome.unlink(missing_ok=True)
