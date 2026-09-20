"""Estrutura de dados para o resultado da validação."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResultadoValidacao:
    """Resultado da validação de integridade ou esquema dos dados."""

    valido: bool
    mensagens_erro: tuple[str, ...] = field(default_factory=tuple)
    mensagens_aviso: tuple[str, ...] = field(default_factory=tuple)
    metricas: dict[str, float | int | str] = field(default_factory=dict)
