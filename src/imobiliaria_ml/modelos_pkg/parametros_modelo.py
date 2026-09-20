"""Estrutura tipada para passagem de hiperparâmetros aos modelos."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParametrosModelo:
    """Encapsula hiperparâmetros para instanciação de modelos sem Any."""

    valores: dict[str, int | float | str | bool | tuple[int, ...] | None] = field(
        default_factory=dict
    )
