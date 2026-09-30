from app_build.rastreamento_mlflow.contrato_observador import (
    ContratoObservador,
    EventoEstatisticaConcluida,
    EventoHoldoutAvaliado,
    EventoModeloSelecionado,
    EventoNestedCvConcluido,
    EventoRegrasNegocioConcluido,
    EventoTreinoFinalConcluido,
)


class DespachanteEventos:
    def __init__(self) -> None:
        self._observadores: list[ContratoObservador] = []

    def registrar_observador(self, observador: ContratoObservador) -> None:
        self._observadores.append(observador)

    def despachar_nested_cv(self, evento: EventoNestedCvConcluido) -> None:
        for obs in self._observadores:
            obs.ao_concluir_nested_cv(evento)

    def despachar_estatistica(self, evento: EventoEstatisticaConcluida) -> None:
        for obs in self._observadores:
            obs.ao_concluir_estatistica(evento)

    def despachar_selecao(self, evento: EventoModeloSelecionado) -> None:
        for obs in self._observadores:
            obs.ao_selecionar_modelo(evento)

    def despachar_treino_final(self, evento: EventoTreinoFinalConcluido) -> None:
        for obs in self._observadores:
            obs.ao_concluir_treino_final(evento)

    def despachar_holdout(self, evento: EventoHoldoutAvaliado) -> None:
        for obs in self._observadores:
            obs.ao_avaliar_holdout(evento)

    def despachar_regras_negocio(self, evento: EventoRegrasNegocioConcluido) -> None:
        for obs in self._observadores:
            obs.ao_concluir_regras_negocio(evento)
